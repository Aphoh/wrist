//! Code for parsing a serialized FX graph.
//!
//!

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

use crate::{
    graph::{ComputeGraph, GraphNode, Subgraph},
    kernels::Kernel,
    network::{Collective, CollectiveType},
};
use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use protobuf::{Message, MessageField};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};
use torch_titan::{NodeData, NodeValue, TensorInfo, TraceResult};

pub struct TraceBuilder {
    trace_result: TraceResult,
    processed: BTreeMap<String, NodeData>,
}

fn filter_node_refs<'a>(nvs: impl Iterator<Item = &'a NodeValue>) -> impl Iterator<Item = &'a str> {
    nvs.filter(|t| t.has_node_ref_value())
        .map(|t| t.node_ref_value())
}

impl TraceBuilder {
    pub fn parse(file: &mut dyn Read) -> Result<Self> {
        let trace_result = TraceResult::parse_from_reader(file)?;
        let processed = Default::default();
        Ok(TraceBuilder {
            trace_result,
            processed,
        })
    }

    fn tensor_info_from_ref(&self, node_ref: &str) -> Result<TensorInfo> {
        self.processed
            .get(node_ref)
            .with_context(|| format!("Node ref '{}' not found", node_ref))
            .and_then(|t| {
                if t.tensor_info.is_none() {
                    return Err(anyhow!("Node {} is not a tensor", t.name));
                }
                Ok(t.tensor_info.clone().unwrap())
            })
    }

    fn tensor_info_from_arg(&self, nd: &NodeData, arg: usize) -> Result<TensorInfo> {
        if nd.args.len() <= arg {
            return Err(anyhow!(
                "Node {} arg index {}>={}",
                nd.name,
                arg,
                nd.args.len()
            ));
        }
        if !nd.args[arg].has_node_ref_value() {
            return Err(anyhow!("Node {} arg {} is not a node_ref", nd.name, arg));
        }
        self.tensor_info_from_ref(nd.args[arg].node_ref_value())
            .with_context(|| format!("Failed to get tensor info from {} arg {}", nd.name, arg))
    }

    fn get_node_ref<'a>(&'a self, nv: &NodeValue) -> Option<&'a NodeData> {
        if !nv.has_node_ref_value() {
            return None;
        }
        self.processed.get(nv.node_ref_value())
    }
    fn parse_call_fn(&self, nd: &NodeData) -> Result<GraphNode> {
        if let Some(node) = self.handle_collective_ops(nd)? {
            return Ok(node);
        }
        if let Some(node) = self.handle_attention_ops(nd)? {
            return Ok(node);
        }
        if let Some(node) = self.handle_mat_operations(nd)? {
            return Ok(node);
        }

        // Add more specialized handler methods as needed:
        // if let Some(node) = self.handle_conv_ops(nd)? { return Ok(node); }

        // Default case - unknown operation
        Ok(GraphNode::Other(nd.target.clone()))
    }

    fn handle_collective_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> {
        let ctype = match nd.target.as_str() {
            "all_gather_into_tensor.default" | "_allgather_base_.default" => {
                CollectiveType::AllGather
            }
            "all_reduce.default" => CollectiveType::AllReduce,
            "reduce_scatter_tensor.default" => CollectiveType::ReduceScatter,
            "all_to_all_single.default" => CollectiveType::AllToAllSingle,
            _ => return Ok(None),
        };
        if !nd.collective_meta.is_some() {
            return Err(anyhow!(
                "Node {} with target {} has no collective meta",
                nd.name,
                nd.target
            ));
        }
        let piece_bytes = nd.collective_meta.comm_tensor_size.try_into()?;
        let ranks = &nd.collective_meta.group_ranks;
        let group_size = ranks.len() as u32;
        let mut group_stride = 0u32;
        if group_size > 1 {
            group_stride = (ranks[1] - ranks[0]).try_into()?;
        }
        return Ok(Some(GraphNode::Collective {
            collective: Collective {
                name: nd.name.clone(),
                ctype,
                group_stride,
                piece_bytes,
                group_size,
            },
            time_us: 1,
        }));
    }

    fn handle_attention_ops(&self, nd: &NodeData) -> Result<Option<GraphNode>> {
        match nd.target.as_str() {
            "_scaled_dot_product_efficient_attention.default"
            | "_scaled_dot_product_flash_attention.default" => {
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
            "_scaled_dot_product_efficient_attention_backward.default"
            | "_scaled_dot_product_flash_attention_backward.default" => {
                // Earlier arguments here are grad tensors
                let qinfo = self.tensor_info_from_arg(nd, 4)?;
                let kinfo = self.tensor_info_from_arg(nd, 5)?;
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::flash_attention_bwd(
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
            _ => Ok(None),
        }
    }

    fn calculate_bytes_for_node_refs<'a>(&self, node_refs: impl AsRef<[NodeValue]>) -> Result<u64> {
        filter_node_refs(node_refs.as_ref().iter())
            .map(|t| {
                self.tensor_info_from_ref(t).map(|t| {
                    let n_params = t.shape.dims.iter().product::<i64>() * 2;
                    let bytes_per_param = match t.dtype.as_str() {
                        "torch.float32" => 4,
                        "torch.float16" => 2,
                        "torch.bfloat16" => 2,
                        _ => {
                            println!("Unknown dtype {}", t.dtype);
                            2
                        }
                    };
                    (n_params * bytes_per_param) as u64
                })
            })
            .sum::<Result<u64>>()
    }

    fn handle_mat_operations(&self, nd: &NodeData) -> Result<Option<GraphNode>> {
        match nd.target.as_str() {
            "matmul" => {
                let ainfo = self.tensor_info_from_arg(nd, 0)?;
                let binfo = self.tensor_info_from_arg(nd, 1)?;
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::matmul(
                        nd.name.clone(),
                        ainfo.shape.dims[0] as u64,
                        ainfo.shape.dims[1] as u64,
                        binfo.shape.dims[1] as u64,
                    ),
                    time_us: 1,
                }))
            }
            "embedding.default" => {
                let winfo = self.tensor_info_from_arg(nd, 0)?;
                let inds_info = self.tensor_info_from_arg(nd, 1)?;
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::embedding(
                        nd.name.clone(),
                        inds_info.shape.dims.iter().product::<i64>() as _,
                        winfo.shape.dims[0] as _,
                        winfo.shape.dims[1] as _,
                    ),
                    time_us: 1,
                }))
            }
            "embedding_dense_backward.default" => {
                let grad_info = self.tensor_info_from_arg(nd, 0)?;
                let grad_shape = &grad_info.shape.dims;
                let batch_size = grad_shape[..grad_shape.len() - 1].iter().product::<i64>() as u64;
                let dim = grad_shape[grad_shape.len() - 1];
                let num_embeddings = nd.args.get(2).unwrap().int_value() as u64;
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::embedding_bwd(
                        nd.name.clone(),
                        batch_size,
                        num_embeddings as _,
                        dim as _,
                    ),
                    time_us: 1,
                }))
            }
            "_fused_adamw_.default" => {
                let params = nd.args[0].sequence_value();
                let grads = nd.args[1].sequence_value();
                let param_bytes = self.calculate_bytes_for_node_refs(&params.elements)?;
                let grad_bytes = self.calculate_bytes_for_node_refs(&grads.elements)?;
                Ok(Some(GraphNode::Kernel {
                    kernel: Kernel::adamw(nd.name.clone(), param_bytes, grad_bytes),
                    time_us: 1,
                }))
            }
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
        let buffers = &self.trace_result.graph_module.buffers;

        for nd in &self.trace_result.graph_module.graph.nodes {
            println!("Processing node '{}'", nd.name);
            // We need a mutable copy since we pull buffer info from graph.buffers for get_attr nodes
            let mut nd_to_insert = nd.clone();
            // By default insert an 'other' node
            let mut new_node = GraphNode::Other(nd.name.clone());
            match nd.op.as_str() {
                "placeholder" => {
                    // These should be all the inputs of the graph
                    inputs.push(nd.clone())
                }
                "get_attr" => {
                    if let Some(info) = buffers.get(&nd.target) {
                        nd_to_insert.tensor_info = MessageField::some(info.clone());
                    }
                }
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
            self.processed.insert(nd.name.clone(), nd_to_insert);
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
        let file = File::open("traces/ws8cp2dp2tp2pp1.pb").unwrap();
        let mut reader = BufReader::new(file);
        let compute_graph = TraceBuilder::parse(&mut reader)
            .unwrap()
            .build_compute_graph()
            .unwrap();
    }
}
