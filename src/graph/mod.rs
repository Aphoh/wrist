use std::collections::BTreeMap;

use petgraph::{graph::NodeIndex as PetNodeIndex, Directed, Direction, Graph};

use crate::{
    kernels::{Kernel, KernelProfile},
    network::{Collective, Network},
};

pub mod fx;

pub enum GraphNode {
    Kernel {
        kernel: Kernel,
        time_us: u64,
    },
    Collective {
        collective: Collective,
        time_us: u64,
    },
    Start,
    End,
}

impl GraphNode {
    pub fn time_us(&self) -> u64 {
        match self {
            GraphNode::Kernel { time_us, .. } => *time_us,
            GraphNode::Collective { time_us, .. } => *time_us,
            GraphNode::Start => 0,
            GraphNode::End => 0,
        }
    }
    pub fn set_time_us(&mut self, new_time_us: u64) {
        match self {
            GraphNode::Kernel { time_us, .. } => *time_us = new_time_us,
            GraphNode::Collective { time_us, .. } => *time_us = new_time_us,
            _ => {}
        }
    }
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphNode::Kernel { kernel, time_us } => {
                f.debug_struct("Kernel")
                    .field("kernel", kernel)
                    .field("time_us", time_us)
                    .finish()
            }
            GraphNode::Collective { collective, time_us } => {
                f.debug_struct("Collective")
                    .field("collective", collective)
                    .field("time_us", time_us)
                    .finish()
            }
            GraphNode::Start => write!(f, "Start"),
            GraphNode::End => write!(f, "End"),
        }
    }
}

type NodeId = PetNodeIndex<u32>;

pub struct DataItem {
    pub name: String,
    pub node_id: NodeId,
}

pub struct Subgraph {
    scan: u64,
    graph: Graph<GraphNode, String, Directed>,
    collective_ids: Vec<NodeId>,
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
                collective_ids: Vec::new(),
            },
            DataItem {
                name: "start".to_string(),
                node_id: start_id,
            },
        )
    }

    pub fn connect(&mut self, edge: impl ToString, from: NodeId, to: NodeId) {
        self.graph.add_edge(from, to, edge.to_string());
    }

    pub fn kernel(
        &mut self,
        from: impl AsRef<[DataItem]>,
        dest_name: impl ToString,
        kernel: Kernel,
        prof: &impl KernelProfile,
    ) -> DataItem {
        let time_us = prof.compute_us(&kernel);
        self.add_node(
            from,
            dest_name,
            GraphNode::Kernel {
                kernel: kernel.clone(),
                time_us,
            },
        )
    }

    pub fn collective(
        &mut self,
        from: impl AsRef<[DataItem]>,
        dest_name: impl ToString,
        collective: Collective,
        prof: &impl Network,
    ) -> DataItem {
        let time_us = prof.measure_one(&collective);
        self.add_node(
            from,
            dest_name,
            GraphNode::Collective {
                collective,
                time_us,
            },
        )
    }

    pub fn add_node(
        &mut self,
        from: impl AsRef<[DataItem]>,
        dest_name: impl ToString,
        node: GraphNode,
    ) -> DataItem {
        let node_id = self.graph.add_node(node);
        if let GraphNode::Collective { .. } = &self.graph[node_id] {
            self.collective_ids.push(node_id);
        }
        for from_di in from.as_ref() {
            self.graph
                .add_edge(from_di.node_id, node_id, from_di.name.clone());
        }
        DataItem {
            name: dest_name.to_string(),
            node_id,
        }
    }

    pub fn remeasure_network(
        &mut self,
        network: &impl Network,
    ) {
        for node_id in self.collective_ids.iter() {
            if let GraphNode::Collective { collective, .. } = &mut self.graph[*node_id] {
                let time_us = network.measure_one(collective);
                self.graph[*node_id].set_time_us(time_us)
            }
        }
    }

    pub fn finish(&mut self, outputs: impl AsRef<[DataItem]>) {
        if self.graph.neighbors_directed(self.start_id, Direction::Outgoing).count() == 0 {
            panic!("Subgraph has non start nodes"); // TODO: better error handling
        }
        if self.graph.neighbors_directed(self.end_id, Direction::Incoming).count() > 0 {
            panic!("Subgraph already finished"); // TODO: better error handling
        }
        if outputs.as_ref().is_empty() {
            panic!("Must provide end nodes"); // TODO: better error handling
        }

        for node in outputs.as_ref() {
            self.graph
                .add_edge(node.node_id, self.end_id, node.name.clone());
        }
    }
    
    fn is_finished(&self) -> bool {
        self.graph.neighbors_directed(self.end_id, Direction::Incoming).count() > 0
    }
}

pub struct ComputeGraph {
    pub subgraphs: Vec<Subgraph>,
}

impl ComputeGraph {
    pub fn new(subgraphs: Vec<Subgraph>) -> Self {
        ComputeGraph { subgraphs }
    }

    pub fn time(&self) -> u64 {
        let mut total_time = 0;
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
                max_time.insert(node_id, input_time + node.time_us());
            }
            total_time += max_time.get(&subgraph.end_id).expect("Must hit end node") * subgraph.scan;
        }
        total_time
    }
}
