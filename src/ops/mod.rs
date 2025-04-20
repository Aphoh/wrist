pub mod double_linear;
pub use double_linear::MLP;

pub mod composite;
pub mod scan;

pub mod attention;
pub use attention::*;

use crate::kernels::{KernelProfile, NamedKernel};
use crate::network::{NamedCollective, Network};
use crate::sharding::{SeqModelSpec, ShardStrategy};
use crate::utils::ValidationError;
use petgraph::graph::{DiGraph, NodeIndex};

#[derive(Default)]
pub struct MemoryProfile {
    pub param_sizes: Vec<u64>,
    pub activation_size: u64,
    pub cache_for_backprop: u64,
    pub input_act_size: u64,
    pub output_act_size: u64,
}

impl MemoryProfile {
    pub fn combine(mut self, next: Self) -> Self {
        self.param_sizes.extend(next.param_sizes);
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

pub enum GraphOp {
    Input,
    Output,
    Barrier,
    Kernel(NamedKernel),
    Collective(NamedCollective),
}
impl GraphOp {
    fn measure(&self, net: &impl Network, kprof: &impl KernelProfile) -> u64 {
        match self {
            GraphOp::Input => 0,
            GraphOp::Output => 0,
            GraphOp::Barrier => 0,
            GraphOp::Kernel(k) => kprof.compute_us(k.kernel),
            GraphOp::Collective(c) => net.measure_one(&c.collective),
        }
    }
}

#[derive(Default)]
pub struct ComputeGraph {
    name: String,
    graph: DiGraph<GraphOp, ()>,
    input_idx: NodeIndex,
}

pub struct ComputeUnit2 {
    graph: ComputeGraph,
    output_idx: NodeIndex,
}

impl ComputeGraph {
    pub fn new(name: &impl ToString) -> (Self, NodeIndex) {
        let mut graph = DiGraph::new();
        let input_idx = graph.add_node(GraphOp::Input);
        (
            Self {
                name: name.to_string(),
                graph,
                input_idx,
            },
            input_idx,
        )
    }

    pub fn kernel(&mut self, deps: impl AsRef<[NodeIndex]>, kernel: NamedKernel) -> NodeIndex {
        self.add_op(deps, GraphOp::Kernel(kernel))
    }

    pub fn maybe_coll(&mut self, dep: NodeIndex, coll: Option<NamedCollective>) -> NodeIndex {
        coll.map(|c| self.add_op(&[dep], GraphOp::Collective(c)))
            .unwrap_or(dep)
    }

    pub fn coll(&mut self, deps: impl AsRef<[NodeIndex]>, coll: NamedCollective) -> NodeIndex {
        self.add_op(deps, GraphOp::Collective(coll))
    }

    pub fn single(
        &mut self,
        deps: impl AsRef<[NodeIndex]>,
        kernel: NamedKernel,
        coll: Option<NamedCollective>,
    ) -> NodeIndex {
        let k_idx = self.graph.add_node(GraphOp::Kernel(kernel));
        let out_idx = self.graph.add_node(GraphOp::Barrier);
        for &dep in deps.as_ref() {
            self.graph.add_edge(dep, k_idx, ());
        }
        self.graph.add_edge(k_idx, out_idx, ());
        coll.map(|c| {
            let c_idx = self.graph.add_node(GraphOp::Collective(c));
            for &dep in deps.as_ref() {
                self.graph.add_edge(dep, c_idx, ());
            }
            self.graph.add_edge(c_idx, out_idx, ());
        });
        out_idx
    }

    pub fn add_op(&mut self, deps: impl AsRef<[NodeIndex]>, op: GraphOp) -> NodeIndex {
        let this_idx = self.graph.add_node(op);
        for &dep in deps.as_ref() {
            self.graph.add_edge(dep, this_idx, ());
        }
        this_idx
    }

    pub fn finish(mut self, deps: impl AsRef<[NodeIndex]>) -> ComputeUnit2 {
        let finish_idx = self.graph.add_node(GraphOp::Output);
        for &dep in deps.as_ref() {
            self.graph.add_edge(dep, finish_idx, ());
        }
        return ComputeUnit2 {
            graph: self,
            output_idx: finish_idx,
        };
    }

    //pub fn measure(&self, net: &impl Network, kprof: &impl KernelProfile) -> u64 {
    //    let tsort = petgraph::algo::toposort(&self.graph, None).unwrap();
    //    let mut longest_path: Vec<i64> = vec![i64::MIN; self.graph.node_count()];
    //    longest_path[self.input_idx.index()] = 0;
    //    let mut prev_idxs = vec![Default::default(); self.graph.node_count()];
    //    for node in tsort {
    //        let mut max_path = 0;
    //        let mut max_idx = Default::default();
    //        for edge in self
    //            .graph
    //            .edges_directed(node, petgraph::Direction::Outgoing)
    //        {
    //            let dst= edge.source();
    //            let measured = self.graph[src].measure(net, kprof);
    //            let path = longest_path[src.index()] + measured;
    //            max_path = max_path.max(path);
    //            max_idx = src;
    //        }
    //        longest_path[node.index()] = max_path;
    //        prev_idxs[node.index()] = max_idx;
    //    }

    //    let mut critical_path = vec![];
    //    let mut node = prev_idxs[self.output_idx.unwrap().index()];
    //    while node != self.input_idx {
    //        critical_path.push(node);
    //        node = prev_idxs[node.index()];
    //    }
    //    let v = longest_path[self.output_idx.unwrap().index()];
    //    v
    //}
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
        is_first_step_in_batch: bool,
    ) -> (Vec<ComputeUnit>, Option<NamedCollective>);

    fn backward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        upstream_collective: Option<NamedCollective>, // This is normally an all-reduce of the previous step's gradients
        is_last_step_in_batch: bool,
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
