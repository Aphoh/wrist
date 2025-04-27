use crate::{
    graph::ComputeGraph,
    kernels::KernelProfile,
    network::Network,
    sharding::{SeqModelSpec, ShardStrategy},
};

pub trait Traceable {
    fn trace(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        kernel_profile: &impl KernelProfile,
    ) -> ComputeGraph;
}
