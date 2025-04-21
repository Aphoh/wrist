use crate::{kernels::KernelProfile, network::Network};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Insufficient memory: {0} < {1}")]
    InsufficientMemory(u64, u64),
    #[error("Cannot split {0} layers into {1} stages")]
    InvalidLayerSplit(u64, u64),
    #[error("Attention: Cannot split {0} q heads into {1} tp groups")]
    InvalidQHeadSplit(u64, u64),
    #[error("Attention: Cannot split {0} kv heads into {1} tp groups")]
    InvalidKVHeadSplit(u64, u64),
    #[error("Batch size {0} is not divisible by {1}")]
    InvalidBatchSplit(u64, u64),
    #[error("Sequence length {0} is not divisible by {1}")]
    InvalidSequenceSplit(u64, u64),
    #[error("MLP intermediate size {0} is not divisible by {1}")]
    InvalidMLPSplit(u64, u64),
    #[error("Invalid feature split: {0} is not divisible by {1}")]
    InvalidFeatureSplit(u64, u64),
}
