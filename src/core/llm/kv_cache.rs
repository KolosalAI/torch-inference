#![allow(dead_code)]
/// PagedAttention KV cache — fixed-size memory blocks for LLM inference.
///
/// # Design
///
/// Instead of allocating contiguous memory per sequence (which causes
/// fragmentation), we divide GPU/CPU memory into fixed-size `Block`s of
/// `block_size` tokens each.  Every sequence gets a `BlockTable` — a list of
/// logical→physical block mappings.  When a sequence needs more tokens it
/// allocates the next free physical block.
///
/// Benefits vs dynamic allocation:
/// - O(1) alloc / free (pointer swap in the free-list)
/// - Zero fragmentation — every block is the same size
/// - Copy-on-write prefix sharing (two sequences pointing to the same prefix
///   block, forked on first write)
use anyhow::Result;
use std::collections::VecDeque;

// ── Types ─────────────────────────────────────────────────────────────────

/// Physical block identifier (index into the pool's backing array).
pub type BlockId = u32;

/// A logical block slot within a sequence's KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRef {
    pub physical_id: BlockId,
    /// Number of tokens currently stored in this block.
    pub num_tokens: usize,
    /// Reference count — how many sequences share this physical block.
    pub ref_count: u32,
}

// ── Config ────────────────────────────────────────────────────────────────

/// Parameters for the KV cache block pool.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of physical blocks in the pool.
    pub num_blocks: usize,
    /// Tokens per block (must match the attention kernel's block size).
    pub block_size: usize,
    /// Number of KV heads (key + value each have this many heads).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
}

impl KvCacheConfig {
    /// Bytes required for one physical block (f16 KV tensors).
    pub fn bytes_per_block(&self) -> usize {
        // 2 (K + V) × num_kv_heads × head_dim × block_size × 2 bytes (f16)
        2 * self.num_kv_heads * self.head_dim * self.block_size * 2
    }

    /// Total bytes for the entire pool.
    pub fn total_bytes(&self) -> usize {
        self.num_blocks * self.bytes_per_block()
    }
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            num_blocks: 2048,
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 128,
        }
    }
}

// ── Block pool ────────────────────────────────────────────────────────────

/// Manages allocation and deallocation of physical KV-cache blocks.
pub struct BlockPool {
    config: KvCacheConfig,
    free_blocks: VecDeque<BlockId>,
    /// Reference count per physical block (index = BlockId).
    ref_counts: Vec<u32>,
}

impl BlockPool {
    pub fn new(config: KvCacheConfig) -> Self {
        let n = config.num_blocks;
        let free_blocks = (0..n as u32).collect();
        let ref_counts = vec![0u32; n];
        Self { config, free_blocks, ref_counts }
    }

    /// Number of free (unallocated) blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total blocks in the pool.
    pub fn num_total_blocks(&self) -> usize {
        self.config.num_blocks
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Allocate a new physical block.  Returns `Err` if the pool is full.
    pub fn allocate(&mut self) -> Result<BlockId> {
        self.free_blocks
            .pop_front()
            .map(|id| {
                self.ref_counts[id as usize] = 1;
                id
            })
            .ok_or_else(|| anyhow::anyhow!("KV cache pool exhausted"))
    }

    /// Increment the reference count of a block (for copy-on-write sharing).
    pub fn add_ref(&mut self, id: BlockId) {
        self.ref_counts[id as usize] += 1;
    }

    /// Decrement the reference count of a block.  Frees it when count reaches 0.
    pub fn release(&mut self, id: BlockId) {
        let rc = &mut self.ref_counts[id as usize];
        if *rc > 0 {
            *rc -= 1;
        }
        if *rc == 0 {
            self.free_blocks.push_back(id);
        }
    }

    /// Get the reference count for a block.
    pub fn ref_count(&self, id: BlockId) -> u32 {
        self.ref_counts[id as usize]
    }

    /// True if a block is exclusively owned (ref_count == 1).
    pub fn is_exclusive(&self, id: BlockId) -> bool {
        self.ref_counts[id as usize] == 1
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }
}

// ── Sequence block table ──────────────────────────────────────────────────

/// Maps a sequence's logical KV positions to physical blocks.
#[derive(Debug, Default)]
pub struct SequenceBlockTable {
    blocks: Vec<BlockId>,
    /// Total tokens stored across all blocks.
    num_tokens: usize,
    block_size: usize,
}

impl SequenceBlockTable {
    pub fn new(block_size: usize) -> Self {
        Self { blocks: Vec::new(), num_tokens: 0, block_size }
    }

    /// Total tokens stored.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Number of physical blocks allocated.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Physical block IDs (for the attention kernel).
    pub fn block_ids(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Tokens remaining in the last (partially-full) block.
    pub fn tokens_in_last_block(&self) -> usize {
        if self.num_tokens == 0 {
            return 0;
        }
        let rem = self.num_tokens % self.block_size;
        if rem == 0 { self.block_size } else { rem }
    }

    /// True if the last block still has free slots.
    pub fn last_block_has_space(&self) -> bool {
        self.blocks.is_empty() || self.tokens_in_last_block() < self.block_size
    }

    /// Append a physical block id to the table.
    pub fn push_block(&mut self, id: BlockId) {
        self.blocks.push(id);
    }

    /// Increment the token count (called after appending a token).
    pub fn increment_tokens(&mut self, n: usize) {
        self.num_tokens += n;
    }

    /// Extend the table by `n` new tokens, allocating blocks from `pool` as needed.
    pub fn extend(&mut self, n: usize, pool: &mut BlockPool) -> Result<()> {
        for _ in 0..n {
            if !self.last_block_has_space() || self.blocks.is_empty() {
                let id = pool.allocate()?;
                self.blocks.push(id);
            }
            self.num_tokens += 1;
        }
        Ok(())
    }

    /// Free all physical blocks back to the pool.
    pub fn free_all(&mut self, pool: &mut BlockPool) {
        for &id in &self.blocks {
            pool.release(id);
        }
        self.blocks.clear();
        self.num_tokens = 0;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> KvCacheConfig {
        KvCacheConfig {
            num_blocks: 8,
            block_size: 4,
            num_kv_heads: 2,
            head_dim: 32,
        }
    }

    // ── KvCacheConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_bytes_per_block() {
        let cfg = KvCacheConfig {
            num_blocks: 1,
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 128,
        };
        // 2 * 8 * 128 * 16 * 2 = 65536
        assert_eq!(cfg.bytes_per_block(), 65536);
    }

    #[test]
    fn test_total_bytes() {
        let cfg = KvCacheConfig {
            num_blocks: 4,
            block_size: 16,
            num_kv_heads: 4,
            head_dim: 64,
        };
        // bytes_per_block = 2*4*64*16*2 = 16384; total = 4*16384 = 65536
        assert_eq!(cfg.total_bytes(), cfg.bytes_per_block() * 4);
    }

    #[test]
    fn test_default_config() {
        let cfg = KvCacheConfig::default();
        assert_eq!(cfg.num_blocks, 2048);
        assert_eq!(cfg.block_size, 16);
    }

    // ── BlockPool ─────────────────────────────────────────────────────────

    #[test]
    fn test_pool_initial_free_count() {
        let pool = BlockPool::new(small_config());
        assert_eq!(pool.num_free_blocks(), 8);
        assert_eq!(pool.num_total_blocks(), 8);
    }

    #[test]
    fn test_pool_allocate_decrements_free() {
        let mut pool = BlockPool::new(small_config());
        let id = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 7);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_pool_allocate_all_then_exhausted() {
        let mut pool = BlockPool::new(small_config());
        for _ in 0..8 {
            pool.allocate().unwrap();
        }
        assert!(pool.allocate().is_err());
    }

    #[test]
    fn test_pool_release_returns_block_to_free_list() {
        let mut pool = BlockPool::new(small_config());
        let id = pool.allocate().unwrap();
        pool.release(id);
        assert_eq!(pool.num_free_blocks(), 8);
    }

    #[test]
    fn test_pool_add_ref_prevents_premature_free() {
        let mut pool = BlockPool::new(small_config());
        let id = pool.allocate().unwrap();
        pool.add_ref(id); // ref_count = 2
        pool.release(id); // ref_count = 1 → not freed yet
        assert_eq!(pool.num_free_blocks(), 7);
        pool.release(id); // ref_count = 0 → freed
        assert_eq!(pool.num_free_blocks(), 8);
    }

    #[test]
    fn test_pool_ref_count_tracks_correctly() {
        let mut pool = BlockPool::new(small_config());
        let id = pool.allocate().unwrap();
        assert_eq!(pool.ref_count(id), 1);
        pool.add_ref(id);
        assert_eq!(pool.ref_count(id), 2);
        pool.release(id);
        assert_eq!(pool.ref_count(id), 1);
        assert!(pool.is_exclusive(id));
    }

    #[test]
    fn test_pool_is_exclusive() {
        let mut pool = BlockPool::new(small_config());
        let id = pool.allocate().unwrap();
        assert!(pool.is_exclusive(id));
        pool.add_ref(id);
        assert!(!pool.is_exclusive(id));
    }

    #[test]
    fn test_pool_block_size() {
        let pool = BlockPool::new(small_config());
        assert_eq!(pool.block_size(), 4);
    }

    // ── SequenceBlockTable ────────────────────────────────────────────────

    #[test]
    fn test_block_table_empty_state() {
        let tbl = SequenceBlockTable::new(4);
        assert_eq!(tbl.num_tokens(), 0);
        assert_eq!(tbl.num_blocks(), 0);
        assert!(tbl.last_block_has_space());
    }

    #[test]
    fn test_block_table_extend_allocates_blocks() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(5, &mut pool).unwrap(); // needs 2 blocks (4 + 1)
        assert_eq!(tbl.num_tokens(), 5);
        assert_eq!(tbl.num_blocks(), 2);
        assert_eq!(pool.num_free_blocks(), 6);
    }

    #[test]
    fn test_block_table_extend_fills_current_block_first() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(2, &mut pool).unwrap(); // 1 block, 2 tokens
        tbl.extend(2, &mut pool).unwrap(); // fills 1st block → still 1 block
        assert_eq!(tbl.num_blocks(), 1);
        assert_eq!(tbl.num_tokens(), 4);
    }

    #[test]
    fn test_block_table_extend_exhausts_pool() {
        let mut pool = BlockPool::new(small_config()); // 8 blocks × 4 = 32 tokens max
        let mut tbl = SequenceBlockTable::new(4);
        assert!(tbl.extend(32, &mut pool).is_ok());
        assert!(tbl.extend(1, &mut pool).is_err());
    }

    #[test]
    fn test_block_table_free_all_returns_blocks() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(8, &mut pool).unwrap();
        assert_eq!(pool.num_free_blocks(), 6);
        tbl.free_all(&mut pool);
        assert_eq!(pool.num_free_blocks(), 8);
        assert_eq!(tbl.num_tokens(), 0);
        assert_eq!(tbl.num_blocks(), 0);
    }

    #[test]
    fn test_block_table_tokens_in_last_block() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(6, &mut pool).unwrap(); // block 0 full (4), block 1 partial (2)
        assert_eq!(tbl.tokens_in_last_block(), 2);
    }

    #[test]
    fn test_block_table_full_last_block() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(4, &mut pool).unwrap(); // exactly 1 full block
        assert_eq!(tbl.tokens_in_last_block(), 4);
        assert!(!tbl.last_block_has_space());
    }

    #[test]
    fn test_block_table_block_ids() {
        let mut pool = BlockPool::new(small_config());
        let mut tbl = SequenceBlockTable::new(4);
        tbl.extend(5, &mut pool).unwrap();
        assert_eq!(tbl.block_ids().len(), 2);
    }

    #[test]
    fn test_pool_config_accessible() {
        let pool = BlockPool::new(small_config());
        assert_eq!(pool.config().block_size, 4);
    }
}
