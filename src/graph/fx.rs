//! Code for parsing a serialized FX graph.
//!
//!

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

use crate::graph::{ComputeGraph, GraphNode, NodeId, Subgraph};
use itertools::Itertools;
use protobuf::Message;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Read,
};
use torch_titan::{NodeData, TraceResult};

pub fn parse_trace_result(file: &mut dyn Read) -> protobuf::Result<TraceResult> {
    TraceResult::parse_from_reader(file)
}

pub fn build_compute_graph(trace: &TraceResult) -> ComputeGraph {
    let (mut sg, start) = Subgraph::new(1);

    let mut processed: BTreeMap<String, NodeData> = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut curr_node = start.clone();

    let mut unknowns = BTreeSet::new();

    for nd in &trace.graph_module.graph.nodes {
        processed.insert(nd.name.clone(), nd.clone());
        let mut new_node = GraphNode::Other(nd.name.clone());
        match nd.op.as_str() {
            "placeholder" => {
                // These should be all the inputs of the graph
                inputs.push(nd.clone())
            }
            "get_attr" => {}
            "call_function" => match nd.target.as_str() {
                "mm" => {
                    todo!()
                }
                _ => {
                    unknowns.insert(nd.target.clone());
                }
            },
            "call_module" => {}
            "call_method" => {}
            "output" => {}
            _ => {}
        }
        curr_node = sg.add_node([&curr_node], nd.name.clone(), new_node);
        for unknown in unknowns.iter().sorted() {
            println!("Unknown node type: {}", unknown);
        }
    }
    //trace.graph_module.as

    return ComputeGraph::new(vec![sg])
}

mod test {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_build_compute_graph() {
        let file = File::open("traces/ws8cp1dp1tp8pp1.pb").unwrap();
        let mut reader = BufReader::new(file);
        let trace_result = parse_trace_result(&mut reader).unwrap();
        let compute_graph = build_compute_graph(&trace_result);
    }
}
