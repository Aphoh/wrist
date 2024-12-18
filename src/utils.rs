use crate::{kernels::KernelProfile, network::Network, ops::ComputeUnit};
use serde_json::error;
use thiserror::Error;

pub fn compute_us(
    compute_units: Vec<ComputeUnit>,
    network: &impl Network,
    kernel_profiler: &impl KernelProfile,
) -> u64 {
    let mut total = 0;
    for unit in compute_units {
        let kernel_time: u64 = unit
            .kernels
            .iter()
            .map(|k| kernel_profiler.compute_us(k.kernel))
            .sum();
        let network_time = network.measure_maybe(&unit.collective);
        total += kernel_time.max(network_time);
    }
    total
}

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
