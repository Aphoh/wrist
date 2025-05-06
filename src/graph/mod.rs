use std::collections::{BTreeMap, BTreeSet};

use petgraph::{graph::NodeIndex as PetNodeIndex, Directed, Direction, Graph};

use crate::{
    kernels::{Kernel, KernelProfile},
    network::{Collective, Network},
};

pub mod fx;

pub enum GraphNode {
    Kernel(Kernel),
    Collective(Collective),
    WaitCollective(String),
    Other(String),
    Start,
    End,
}

impl From<Kernel> for GraphNode {
    fn from(kernel: Kernel) -> Self {
        GraphNode::Kernel(kernel)
    }
}

impl From<Collective> for GraphNode {
    fn from(collective: Collective) -> Self {
        GraphNode::Collective(collective)
    }
}

impl GraphNode {
    pub fn time_us<N: Network, K: KernelProfile>(&self, network: &N, kp: &K) -> Option<u64> {
        match self {
            GraphNode::Kernel(k) => kp.compute_us(&k.op),
            GraphNode::Collective(c) => network.measure(c),
            GraphNode::WaitCollective(_) => Some(0),
            GraphNode::Other(_) => Some(0),
            GraphNode::Start => Some(0),
            GraphNode::End => Some(0),
        }
    }
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphNode::Kernel(kernel) => f.debug_struct("Kernel").field("kernel", kernel).finish(),
            GraphNode::Collective(collective) => f
                .debug_struct("Collective")
                .field("collective", collective)
                .finish(),
            GraphNode::WaitCollective(name) => write!(f, "WaitCollective({})", name),
            GraphNode::Other(name) => write!(f, "Other({})", name),
            GraphNode::Start => write!(f, "Start"),
            GraphNode::End => write!(f, "End"),
        }
    }
}

type NodeId = PetNodeIndex<u32>;

#[derive(Debug, Clone)]
pub struct DataItem {
    pub name: String,
    pub node_id: NodeId,
}

impl From<&DataItem> for NodeId {
    fn from(data_item: &DataItem) -> Self {
        data_item.node_id
    }
}

pub struct Subgraph {
    scan: u64,
    graph: Graph<GraphNode, (), Directed>,
    start_id: NodeId,
    end_id: NodeId,
}

impl Subgraph {
    pub fn new(scan: u64) -> (Self, DataItem) {
        let mut graph = Graph::new();
        let start_id = graph.add_node(GraphNode::Start);
        let end_id = graph.add_node(GraphNode::End);
        (
            Subgraph {
                scan,
                graph,
                start_id,
                end_id,
            },
            DataItem {
                name: "start".to_string(),
                node_id: start_id,
            },
        )
    }

    fn add_node<'a, T: 'a>(
        &mut self,
        from: impl AsRef<[&'a T]>,
        dest_name: impl ToString,
        elem: impl Into<GraphNode>,
    ) -> DataItem
    where
        NodeId: From<&'a T>,
    {
        let node = elem.into();
        let node_id = self.graph.add_node(node);
        for &from_di in from.as_ref() {
            self.graph.add_edge(from_di.into(), node_id, ());
        }
        DataItem {
            name: dest_name.to_string(),
            node_id,
        }
    }

    pub fn finish(&mut self, outputs: impl AsRef<[DataItem]>) {
        if self
            .graph
            .neighbors_directed(self.start_id, Direction::Outgoing)
            .count()
            == 0
        {
            panic!("Subgraph has non start nodes"); // TODO: better error handling
        }
        if self
            .graph
            .neighbors_directed(self.end_id, Direction::Incoming)
            .count()
            > 0
        {
            panic!("Subgraph already finished"); // TODO: better error handling
        }
        if outputs.as_ref().is_empty() {
            panic!("Must provide end nodes"); // TODO: better error handling
        }

        for node in outputs.as_ref() {
            self.graph.add_edge(node.node_id, self.end_id, ());
        }
    }

    fn is_finished(&self) -> bool {
        self.graph
            .neighbors_directed(self.end_id, Direction::Incoming)
            .count()
            > 0
    }
}

#[derive(Default, Debug)]
pub struct MissingProfiles {
    pub missing_kernels: BTreeSet<Kernel>,
    pub missing_collectives: BTreeSet<Collective>,
}

impl MissingProfiles {
    pub fn is_empty(&self) -> bool {
        self.missing_kernels.is_empty() && self.missing_collectives.is_empty()
    }
    pub fn merge(&mut self, other: MissingProfiles) {
        self.missing_kernels
            .extend(other.missing_kernels.into_iter());
        self.missing_collectives
            .extend(other.missing_collectives.into_iter());
    }
}

pub struct ComputeGraph {
    pub subgraphs: Vec<Subgraph>,
    pub peak_memory: u64,
}

impl ComputeGraph {
    pub fn new(subgraphs: Vec<Subgraph>, peak_memory: u64) -> Self {
        ComputeGraph { subgraphs, peak_memory }
    }

    pub fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    pub fn time<N: Network, K: KernelProfile>(
        &self,
        net: &N,
        kernel: &K,
    ) -> Result<u64, MissingProfiles> {
        let mut total_time = 0;

        let mut missing_profiles = MissingProfiles::default();
        for subgraph in &self.subgraphs {
            if !subgraph.is_finished() {
                panic!("Subgraph not finished"); // TODO: better error handling
            }
            let mut max_time = BTreeMap::new();
            max_time.insert(subgraph.start_id, 0);
            let sorted = petgraph::algo::toposort(&subgraph.graph, None).expect("No cycles plz");
            for node_id in sorted {
                let node = &subgraph.graph[node_id];
                let mut input_time: u64 = 0;
                for neighbor in subgraph
                    .graph
                    .neighbors_directed(node_id, Direction::Incoming)
                {
                    // Get longest time from neighbor
                    input_time = max_time[&neighbor].max(input_time);
                }
                let time_us = node.time_us(net, kernel).unwrap_or_else(|| {
                    match node {
                        GraphNode::Kernel(k) => missing_profiles.missing_kernels.insert(k.clone()),
                        GraphNode::Collective(c) => {
                            missing_profiles.missing_collectives.insert(c.clone())
                        }
                        _ => true,
                    };
                    0
                });
                max_time.insert(node_id, input_time + time_us);
            }
            total_time +=
                max_time.get(&subgraph.end_id).expect("Must hit end node") * subgraph.scan;
        }
        if missing_profiles.missing_kernels.is_empty()
            && missing_profiles.missing_collectives.is_empty()
        {
            Ok(total_time)
        } else {
            Err(missing_profiles)
        }
    }
}
