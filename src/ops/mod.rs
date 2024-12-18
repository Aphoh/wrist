pub mod double_linear;
pub use double_linear::MLP;

pub mod composite;
pub mod scan;

pub mod attention;
pub use attention::*;

use crate::kernels::NamedKernel;
use crate::network::NamedCollective;
use crate::sharding::{SeqModelSpec, ShardStrategy};
use crate::utils::ValidationError;

#[derive(Default)]
pub struct MemoryProfile {
    pub weight_size: u64,
    pub activation_size: u64,
    pub cache_for_backprop: u64,
    pub input_act_size: u64,
    pub output_act_size: u64,
}

impl MemoryProfile {
    pub fn combine(&self, next: &Self) -> Self {
        Self {
            weight_size: self.weight_size + next.weight_size,
            activation_size: self.activation_size.max(next.activation_size),
            cache_for_backprop: self.cache_for_backprop + next.cache_for_backprop,
            input_act_size: self.input_act_size,
            output_act_size: next.output_act_size,
        }
    }
    pub fn total_mem_usage(&self, dp_size: u64) -> u64 {
        // fp32 adam optimizer states +
        let weight_opt_memory = 6 * self.weight_size + 12 * self.weight_size / dp_size;
        return weight_opt_memory + self.activation_size + self.cache_for_backprop;
    }
}

pub enum GraphNode {
    Kernel(NamedKernel),
    Collective(NamedCollective),
}

#[derive(Default)]
pub struct ComputeGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<(usize, usize)>,
    finish_nodes: Vec<usize>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn kernel(&mut self, deps: impl AsRef<[usize]>, kernel: NamedKernel) -> usize {
        self.add_node(deps, GraphNode::Kernel(kernel))
    }

    pub fn coll(&mut self, deps: impl AsRef<[usize]>, coll: NamedCollective) -> usize {
        self.add_node(deps, GraphNode::Collective(coll))
    }

    pub fn add_node(&mut self, deps: impl AsRef<[usize]>, node: GraphNode) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(node);
        for &dep in deps.as_ref() {
            self.edges.push((dep, node_idx));
        }
        node_idx
    }

    pub fn finish(&mut self, nodes: impl AsRef<[usize]>) {
        self.finish_nodes.extend_from_slice(nodes.as_ref());
    }
}

#[derive(Default)]
pub struct ComputeUnit {
    pub kernels: Vec<NamedKernel>,
    pub collective: Option<NamedCollective>,
}

impl ComputeUnit {
    pub fn new(kernels: Vec<NamedKernel>, collective: Option<NamedCollective>) -> Self {
        ComputeUnit {
            kernels,
            collective,
        }
    }

    pub fn single(kernel: NamedKernel, collective: Option<NamedCollective>) -> Self {
        ComputeUnit {
            kernels: vec![kernel],
            collective: collective,
        }
    }

    pub fn konly(kernel: NamedKernel) -> Self {
        ComputeUnit {
            kernels: vec![kernel],
            collective: None,
        }
    }

    pub fn conly(collective: Option<NamedCollective>) -> Self {
        ComputeUnit {
            kernels: vec![],
            collective,
        }
    }
}

pub trait Operation {
    fn forward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        downstream_collective: Option<NamedCollective>, // This is normally an all-gather that the next layer needs
    ) -> (Vec<ComputeUnit>, Option<NamedCollective>);

    fn backward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        upstream_collective: Option<NamedCollective>, // This is normally an all-reduce of the previous step's gradients
    ) -> (Vec<ComputeUnit>, Option<NamedCollective>);
    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> MemoryProfile;

    fn validate(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
    ) -> Result<(), ValidationError>;
    // TODO: deal with batch network ops
    //fn batch_network_ops(&self, axes: SeqModelSpec, strategy: &ShardStrategy<M>) -> Vec<Collective>;

    fn memory_profile(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> MemoryProfile {
        self.memory_bytes(axes, strategy)
    }
}
