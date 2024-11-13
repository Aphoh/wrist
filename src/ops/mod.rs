pub mod double_linear;
pub mod scan;
pub use double_linear::MLP;

use crate::kernels::Kernel;
use crate::network::Collective;
use crate::sharding::{SeqModelSpec, ShardStrategy};

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

#[derive(Default)]
pub struct ComputeUnit {
    pub kernels: Vec<Kernel>,
    pub collective: Option<Collective>,
}

impl ComputeUnit {
    pub fn new(kernels: Vec<Kernel>, collective: Option<Collective>) -> Self {
        ComputeUnit {
            kernels,
            collective,
        }
    }

    pub fn new_mergeable(kernels: Vec<Kernel>, collective: Option<Collective>) -> Self {
        ComputeUnit {
            kernels,
            collective,
        }
    }
}

pub trait Operation {
    fn forward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        downstream_collective: Option<Collective>,
    ) -> (Vec<ComputeUnit>, Option<Collective>);

    fn backward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        upstream_collective: Option<Collective>,
    ) -> (Vec<ComputeUnit>, Option<Collective>);
    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> MemoryProfile;

    fn validate(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> bool;
    // TODO: deal with batch network ops
    //fn batch_network_ops(&self, axes: SeqModelSpec, strategy: &ShardStrategy<M>) -> Vec<Collective>;
}
