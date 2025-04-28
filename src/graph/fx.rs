//! Code for parsing a serialized FX graph.
//!
//!

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

use crate::{
    graph::{ComputeGraph, GraphNode, Subgraph},
    kernels::Kernel,
    network::Collective,
};
use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use protobuf::Message;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};
use torch_titan::{NodeData, NodeValue, TensorInfo, TraceResult};

pub struct TraceBuilder {
    trace_result: TraceResult,
    processed: BTreeMap<String, NodeData>,
}

impl TraceBuilder {
    pub fn parse(file: &mut dyn Read) -> Result<Self> {
        let trace_result = TraceResult::parse_from_reader(file)?;
        Ok(TraceBuilder {
            trace_result,
            processed: Default::default(),
        })
    }

    fn tensor_info_from_arg(&self, nd: &NodeData, arg: usize) -> Result<TensorInfo> {
        if nd.args.len() <= arg {
            return Err(anyhow!(
                "Node {} with {} args, arg index {} out of bounds",
                nd.name,
                nd.args.len(),
                arg
            ));
        }
        if !nd.args[arg].has_node_ref_value() {
            return Err(anyhow!("Node {} arg {} is not a node_ref", nd.name, arg));
        }
        self.processed
            .get(nd.args[arg].node_ref_value())
            .with_context(|| {
                format!(
                    "Node ref '{}' not found when parsing argument {} of node '{}'",
                    nd.args[arg].node_ref_value(),
                    arg,
                    nd.name
                )
            })
            .and_then(|t| {
                if t.tensor_info.is_none() {
                    return Err(anyhow!(
                        "Node {} arg {} ({}) is not a tensor",
                        nd.name,
                        arg,
                        t.name
                    ));
                }
                Ok(t.tensor_info.clone().unwrap())
            })
    }

    fn get_node_ref<'a>(&'a self, nv: &NodeValue) -> Option<&'a NodeData> {
        if !nv.has_node_ref_value() {
            return None;
        }
        self.processed.get(nv.node_ref_value())
    }
    fn parse_call_fn(&self, nd: &NodeData) -> Result<GraphNode> {
        // Try to handle with our registry of handlers
        if let Some(node) = self.handle_collective_ops(nd)? {
            return Ok(node);
        }
        
        if let Some(node) = self.handle_attention_ops(nd)? {
            return Ok(node);
        }
        
        // Add more specialized handler methods as needed:
        // if let Some(node) = self.handle_conv_ops(nd)? { return Ok(node); }
        
        // Default case - unknown operation
        Ok(GraphNode::Other(nd.target.clone()))
    }
    
    fn handle_collective_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> {
        match nd.target.as_str() {
            "all_gather_into_tensor" => {
                assert!(nd.collective_meta.is_some());
                let piece_bytes = nd.collective_meta.comm_tensor_size.try_into()?;
                let ranks = &nd.collective_meta.group_ranks;
                let group_size = ranks.len() as u32;
                let mut group_stride = 0u32;
                if group_size > 1 {
                    group_stride = (ranks[1] - ranks[0]).try_into()?;
                }
                Ok(Some(GraphNode::Collective {
                    collective: Collective::all_gather(
                        nd.name.clone(),
                        piece_bytes,
                        group_size,
                        group_stride,
                    ),
                    time_us: 0,
                }))
            }
            // Add other collective operations here
            // "all_reduce" => { ... }
            // "reduce_scatter" => { ... }
            _ => Ok(None),
        }
    }
    
    fn handle_attention_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> {
        match nd.target.as_str() {
            "_scaled_dot_product_efficient_attention.default" => {
                // {B, H, S, D}
                let qinfo = self.tensor_info_from_arg(nd, 0)?;
                let kinfo = self.tensor_info_from_arg(nd, 1)?;
                //let kv_info = processed.get(&)
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::flash_attention(
                        nd.name.to_string(),
                        qinfo.shape.dims[0] as u64,
                        qinfo.shape.dims[2] as u64,
                        kinfo.shape.dims[1] as u64,
                        qinfo.shape.dims[1] as u64,
                        qinfo.shape.dims[3] as u64,
                    ),
                    time_us: 1,
                }))
            }
            // Add other attention operations here
            // "self_attention" => { ... }
            // "cross_attention" => { ... }
            _ => Ok(None),
        }
    }
    
    // Add more handler methods for different categories:
    // fn handle_conv_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> { ... }
    // fn handle_linear_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> { ... }

    pub fn build_compute_graph(&mut self) -> Result<ComputeGraph> {
        let (mut sg, start) = Subgraph::new(1);

        let mut inputs = Vec::new();
        let mut curr_node = start.clone();

        let mut unknowns = BTreeSet::new();

        for nd in &self.trace_result.graph_module.graph.nodes {
            println!("Processing node '{}'", nd.name);
            self.processed.insert(nd.name.clone(), nd.clone());
            let mut new_node = GraphNode::Other(nd.name.clone());
            match nd.op.as_str() {
                "placeholder" => {
                    // These should be all the inputs of the graph
                    inputs.push(nd.clone())
                }
                "get_attr" => {}
                "call_function" => {
                    new_node = self.parse_call_fn(&nd)?;
                    if let GraphNode::Other(name) = &new_node {
                        unknowns.insert(name.clone());
                    }
                }
                "call_module" => {}
                "call_method" => {}
                "output" => {}
                _ => {
                    println!("Unknown op {} in node {}", nd.op, nd.name);
                }
            }
            curr_node = sg.add_node([&curr_node], nd.name.clone(), new_node);
        }
        println!("Unknown call functions:");
        for unk in unknowns.iter().sorted() {
            println!("{}", unk);
        }
        //trace.graph_module.as

        Ok(ComputeGraph::new(vec![sg]))
    }
}

mod test {
    use super::TraceBuilder;
    use std::fs::File;
    use std::io::BufReader;
    #[test]
    fn test_build_compute_graph() {
        let file = File::open("traces/ws8cp1dp1tp8pp1.pb").unwrap();
        let mut reader = BufReader::new(file);
        let compute_graph = TraceBuilder::parse(&mut reader)
            .unwrap()
            .build_compute_graph()
            .unwrap();
    }
}
