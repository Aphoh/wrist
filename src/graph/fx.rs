//! Code for parsing a serialized FX graph.
//! 
//! 

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

use std::io::Read;

use protobuf::Message;

use torch_titan::TraceResult;

pub fn parse_trace_result(file: &mut dyn Read) -> protobuf::Result<TraceResult> {
    TraceResult::parse_from_reader(file)
}