/// Continuous batching scheduler — iteration-level sequence management.
///
/// Unlike fixed-timeout batching (where the server waits for N requests or T ms
/// before running a forward pass), continuous batching dispatches *every* step
/// with all currently runnable sequences.  New sequences are admitted to the
/// running pool as soon as there is KV-cache capacity.
///
/// # States
///
/// ```text
/// WAITING ──admit──► RUNNING ──finish──► FINISHED
///                      │
///                   preempt
///                      │
///                   WAITING (re-queued with KV freed)
/// ```
use std::collections::VecDeque;

use crate::core::llm::kv_cache::{BlockPool, KvCacheConfig, SequenceBlockTable};

// ── Sequence ──────────────────────────────────────────────────────────────

/// Unique identifier for an in-flight sequence.
pub type SequenceId = u64;

/// Status of a sequence in the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqStatus {
    Waiting,
    Running,
    Finished,
    Preempted,
}

/// An LLM generation sequence.
pub struct Sequence {
    pub id: SequenceId,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub status: SeqStatus,
    pub max_tokens: usize,
    pub block_table: SequenceBlockTable,
}

impl Sequence {
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>, max_tokens: usize, block_size: usize) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            status: SeqStatus::Waiting,
            max_tokens,
            block_table: SequenceBlockTable::new(block_size),
        }
    }

    /// Total tokens (prompt + output).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// True if the sequence has reached its token limit.
    pub fn is_finished(&self) -> bool {
        self.output_tokens.len() >= self.max_tokens
    }

    /// Append a generated token.
    pub fn append_token(&mut self, token: u32) {
        self.output_tokens.push(token);
    }
}

// ── Scheduler config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum sequences running simultaneously.
    pub max_num_seqs: usize,
    /// Maximum total tokens in a single forward-pass batch (prompt + decode).
    pub max_num_batched_tokens: usize,
    /// KV cache configuration.
    pub kv_cache: KvCacheConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            kv_cache: KvCacheConfig::default(),
        }
    }
}

// ── Scheduler output ──────────────────────────────────────────────────────

/// What the engine needs to process in one forward pass.
#[derive(Debug)]
pub struct SchedulerOutput {
    /// Sequences running in this step (prefill or decode).
    pub running_seq_ids: Vec<SequenceId>,
    /// Sequences newly admitted from the waiting queue (need prefill).
    pub prefill_seq_ids: Vec<SequenceId>,
    /// Total tokens to process.
    pub num_batched_tokens: usize,
}

// ── Scheduler ─────────────────────────────────────────────────────────────

/// Manages sequence lifecycle and KV-cache allocation for continuous batching.
pub struct Scheduler {
    config: SchedulerConfig,
    pool: BlockPool,
    waiting: VecDeque<SequenceId>,
    running: Vec<SequenceId>,
    sequences: std::collections::HashMap<SequenceId, Sequence>,
    next_id: SequenceId,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let pool = BlockPool::new(config.kv_cache.clone());
        Self {
            config,
            pool,
            waiting: VecDeque::new(),
            running: Vec::new(),
            sequences: std::collections::HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a new sequence to the waiting queue.  Returns its assigned id.
    pub fn add_sequence(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;
        let block_size = self.pool.block_size();
        let seq = Sequence::new(id, prompt_tokens, max_tokens, block_size);
        self.sequences.insert(id, seq);
        self.waiting.push_back(id);
        id
    }

    /// Compute the next batch to process.
    ///
    /// Called once per forward-pass iteration.  Attempts to admit sequences
    /// from `waiting` into `running` subject to capacity constraints, then
    /// returns all running sequences.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut prefill_seq_ids: Vec<SequenceId> = Vec::new();
        let mut num_batched_tokens = 0usize;

        // Try to admit waiting sequences.
        while let Some(&candidate_id) = self.waiting.front() {
            let seq = match self.sequences.get(&candidate_id) {
                Some(s) => s,
                None => { self.waiting.pop_front(); continue; }
            };

            let prompt_len = seq.prompt_tokens.len();
            let would_add = prompt_len;

            // Capacity checks.
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }
            if num_batched_tokens + would_add > self.config.max_num_batched_tokens {
                break;
            }
            let blocks_needed = (prompt_len + self.pool.block_size() - 1) / self.pool.block_size();
            if self.pool.num_free_blocks() < blocks_needed {
                break; // Not enough KV cache; keep waiting.
            }

            // Admit.
            self.waiting.pop_front();
            let seq = self.sequences.get_mut(&candidate_id).unwrap();
            seq.status = SeqStatus::Running;
            // Allocate KV blocks for the prompt.
            seq.block_table
                .extend(prompt_len, &mut self.pool)
                .expect("block allocation failed after free-block check");
            num_batched_tokens += prompt_len;
            self.running.push(candidate_id);
            prefill_seq_ids.push(candidate_id);
        }

        // Add decode steps for already-running sequences.
        for &id in &self.running {
            if !prefill_seq_ids.contains(&id) {
                num_batched_tokens += 1; // one decode token per step
                // Allocate one token worth of KV space.
                if let Some(seq) = self.sequences.get_mut(&id) {
                    let _ = seq.block_table.extend(1, &mut self.pool);
                }
            }
        }

        SchedulerOutput {
            running_seq_ids: self.running.clone(),
            prefill_seq_ids,
            num_batched_tokens,
        }
    }

    /// Record that sequence `id` produced token `tok`.
    ///
    /// Marks the sequence finished if it hit `max_tokens` or produced a stop
    /// token (token id 2 = EOS by convention here).
    pub fn append_token(&mut self, id: SequenceId, tok: u32) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.append_token(tok);
            if seq.is_finished() || tok == 2 {
                seq.status = SeqStatus::Finished;
            }
        }
    }

    /// Remove all finished sequences from the running pool, freeing their KV blocks.
    ///
    /// Returns the ids of sequences that were removed.
    pub fn drain_finished(&mut self) -> Vec<SequenceId> {
        let mut finished = Vec::new();
        self.running.retain(|&id| {
            let is_done = self.sequences
                .get(&id)
                .map(|s| s.status == SeqStatus::Finished)
                .unwrap_or(true);
            if is_done {
                finished.push(id);
                false
            } else {
                true
            }
        });
        // Free KV blocks.
        for &id in &finished {
            if let Some(seq) = self.sequences.get_mut(&id) {
                seq.block_table.free_all(&mut self.pool);
            }
        }
        finished
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn num_waiting(&self) -> usize { self.waiting.len() }
    pub fn num_running(&self) -> usize { self.running.len() }
    pub fn num_free_blocks(&self) -> usize { self.pool.num_free_blocks() }

    pub fn get_sequence(&self, id: SequenceId) -> Option<&Sequence> {
        self.sequences.get(&id)
    }

    pub fn config(&self) -> &SchedulerConfig { &self.config }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::kv_cache::KvCacheConfig;

    fn small_scheduler() -> Scheduler {
        Scheduler::new(SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            kv_cache: KvCacheConfig {
                num_blocks: 16,
                block_size: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
        })
    }

    // ── Sequence ──────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_new_state() {
        let seq = Sequence::new(1, vec![1, 2, 3], 10, 4);
        assert_eq!(seq.id, 1);
        assert_eq!(seq.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(seq.status, SeqStatus::Waiting);
        assert_eq!(seq.total_tokens(), 3);
        assert!(!seq.is_finished());
    }

    #[test]
    fn test_sequence_append_token() {
        let mut seq = Sequence::new(1, vec![1, 2], 3, 4);
        seq.append_token(99);
        assert_eq!(seq.output_tokens, vec![99]);
        assert_eq!(seq.total_tokens(), 3);
    }

    #[test]
    fn test_sequence_is_finished_when_max_tokens_reached() {
        let mut seq = Sequence::new(1, vec![1], 2, 4);
        seq.append_token(10);
        seq.append_token(11);
        assert!(seq.is_finished());
    }

    // ── Scheduler ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_sequence_increments_waiting() {
        let mut sched = small_scheduler();
        sched.add_sequence(vec![1, 2, 3], 10);
        assert_eq!(sched.num_waiting(), 1);
        assert_eq!(sched.num_running(), 0);
    }

    #[test]
    fn test_schedule_admits_waiting_sequence() {
        let mut sched = small_scheduler();
        let id = sched.add_sequence(vec![1, 2, 3], 10);
        let out = sched.schedule();
        assert_eq!(sched.num_waiting(), 0);
        assert_eq!(sched.num_running(), 1);
        assert!(out.prefill_seq_ids.contains(&id));
        assert!(out.running_seq_ids.contains(&id));
    }

    #[test]
    fn test_schedule_respects_max_num_seqs() {
        let mut sched = small_scheduler();
        for _ in 0..6 {
            sched.add_sequence(vec![1], 5);
        }
        sched.schedule();
        // max_num_seqs = 4 → only 4 admitted
        assert_eq!(sched.num_running(), 4);
        assert_eq!(sched.num_waiting(), 2);
    }

    #[test]
    fn test_schedule_decode_step_increments_tokens() {
        let mut sched = small_scheduler();
        sched.add_sequence(vec![1, 2], 10);
        let out1 = sched.schedule(); // prefill step
        sched.append_token(*out1.running_seq_ids.first().unwrap(), 100);
        let out2 = sched.schedule(); // decode step
        assert_eq!(out2.prefill_seq_ids.len(), 0);
        assert_eq!(out2.running_seq_ids.len(), 1);
    }

    #[test]
    fn test_append_token_and_finish() {
        let mut sched = small_scheduler();
        let id = sched.add_sequence(vec![1], 2);
        sched.schedule();
        sched.append_token(id, 10);
        sched.append_token(id, 11); // hits max_tokens = 2
        let finished = sched.drain_finished();
        assert!(finished.contains(&id));
        assert_eq!(sched.num_running(), 0);
    }

    #[test]
    fn test_eos_token_finishes_sequence() {
        let mut sched = small_scheduler();
        let id = sched.add_sequence(vec![1], 100);
        sched.schedule();
        sched.append_token(id, 2); // EOS
        let finished = sched.drain_finished();
        assert!(finished.contains(&id));
    }

    #[test]
    fn test_drain_finished_frees_kv_blocks() {
        let mut sched = small_scheduler();
        let free_before = sched.num_free_blocks();
        sched.add_sequence(vec![1, 2, 3, 4], 2);
        sched.schedule(); // allocates KV blocks
        let free_after_alloc = sched.num_free_blocks();
        assert!(free_after_alloc < free_before);
        // Finish the sequence.
        let seq_id = *sched.running.first().unwrap();
        sched.append_token(seq_id, 10);
        sched.append_token(seq_id, 11);
        sched.drain_finished();
        // Blocks should be back.
        assert_eq!(sched.num_free_blocks(), free_before);
    }

    #[test]
    fn test_get_sequence_returns_correct_seq() {
        let mut sched = small_scheduler();
        let id = sched.add_sequence(vec![1, 2, 3], 10);
        let seq = sched.get_sequence(id).unwrap();
        assert_eq!(seq.prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_sequence_nonexistent_returns_none() {
        let sched = small_scheduler();
        assert!(sched.get_sequence(9999).is_none());
    }

    #[test]
    fn test_scheduler_config_accessible() {
        let sched = small_scheduler();
        assert_eq!(sched.config().max_num_seqs, 4);
    }

    // ── SeqStatus ─────────────────────────────────────────────────────────

    #[test]
    fn test_seq_status_equality() {
        assert_eq!(SeqStatus::Waiting, SeqStatus::Waiting);
        assert_ne!(SeqStatus::Running, SeqStatus::Finished);
    }

    #[test]
    fn test_seq_status_clone_debug() {
        let s = SeqStatus::Running;
        let c = s.clone();
        assert_eq!(s, c);
        assert!(format!("{:?}", c).contains("Running"));
    }
}
