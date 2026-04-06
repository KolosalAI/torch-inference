pub mod kv_cache;
/// LLM inference subsystem.
///
/// # Components
///
/// | Module | Role |
/// |--------|------|
/// | `sampler` | Token sampling: greedy, temperature, top-k, top-p |
/// | `kv_cache` | PagedAttention KV cache: fixed-size block pool |
/// | `scheduler` | Continuous batching: iteration-level sequence management |
/// | `speculative` | Speculative decoding: draft+target model pipeline |
pub mod sampler;
pub mod scheduler;
pub mod speculative;

// Re-export the most-used types.
#[allow(unused_imports)]
pub use kv_cache::{BlockPool, KvCacheConfig, SequenceBlockTable};
#[allow(unused_imports)]
pub use sampler::SamplingParams;
#[allow(unused_imports)]
pub use scheduler::{Scheduler, SchedulerConfig, SeqStatus, Sequence, SequenceId};
#[allow(unused_imports)]
pub use speculative::{DraftModel, SpecConfig, SpeculativeDecoder, TargetModel};

#[cfg(feature = "candle")]
pub mod candle_backend;
#[cfg(feature = "candle")]
pub use candle_backend::CandleLlmBackend;
