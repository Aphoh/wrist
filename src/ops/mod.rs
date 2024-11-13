
pub mod double_linear;
pub mod scan;
pub use double_linear::DoubleLinearReductionParallel;

use crate::sharding::{ShardStrategy, SeqModelSpec};
use crate::network::Collective;

#[derive(Default)]
pub struct MemoryProfile {
    pub weight_memory: u64,
    pub activation_memory: u64,
    pub cache_for_backprop: u64,
    pub gradient_size: u64,
}

impl MemoryProfile {
    pub fn combine(&self, other: &Self) -> Self {
        Self {
            weight_memory: self.weight_memory + other.weight_memory,
            activation_memory: self.activation_memory,
            cache_for_backprop: self.cache_for_backprop + other.cache_for_backprop,
            gradient_size: self.gradient_size + other.gradient_size,
        }
    }
    pub fn total(&self) -> u64 {
        self.weight_memory + self.activation_memory + self.cache_for_backprop + self.gradient_size
    }
}

pub trait Operation {
    fn forward_us(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> u64;
    // If the operation isn't just 2*forward, we should return a different value here
    fn backward_us(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> Option<u64>;
    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> MemoryProfile;
    fn micro_batch_fwd_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
    ) -> Vec<Collective>;
    fn micro_batch_bwd_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
    ) -> Vec<Collective>;

    fn validate(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> bool;
    // TODO: deal with batch network ops
    //fn batch_network_ops(&self, axes: SeqModelSpec, strategy: &ShardStrategy<M>) -> Vec<Collective>;
}