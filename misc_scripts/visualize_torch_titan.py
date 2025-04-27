#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the generated protobuf classes
import src.graph.torch_titan_pb2 as pb

def format_node_value(node_value):
    """Format a NodeValue into a readable string."""
    which_oneof = node_value.WhichOneof('value_oneof')
    
    if which_oneof == 'null_value':
        return 'None'
    elif which_oneof == 'bool_value':
        return str(node_value.bool_value)
    elif which_oneof == 'int_value':
        return str(node_value.int_value)
    elif which_oneof == 'float_value':
        return str(node_value.float_value)
    elif which_oneof == 'string_value':
        return f'"{node_value.string_value}"'
    elif which_oneof == 'device_value':
        return f'device="{node_value.device_value}"'
    elif which_oneof == 'dtype_value':
        return f'dtype={node_value.dtype_value}'
    elif which_oneof == 'shape_value':
        dims = node_value.shape_value.dims
        return f"({', '.join(str(d) for d in dims)})" if dims else "()"
    elif which_oneof == 'sequence_value':
        elements = [format_node_value(elem) for elem in node_value.sequence_value.elements]
        return f"[{', '.join(elements)}]"
    elif which_oneof == 'node_ref_value':
        return node_value.node_ref_value
    elif which_oneof == 'repr_value':
        return node_value.repr_value
    else:
        return '<unknown>'

def format_args(args):
    """Format a list of NodeValue args."""
    return ', '.join(format_node_value(arg) for arg in args)

def format_kwargs(kwargs):
    """Format a list of NamedNodeValue kwargs."""
    return ', '.join(f"{kw.name}={format_node_value(kw.value)}" for kw in kwargs)

def format_tensor_info(tensor_info: pb.TensorInfo) -> List[str]:
    """Format TensorInfo for display."""
    shape = ', '.join(str(d) for d in tensor_info.shape.dims)
    dtype = tensor_info.dtype
    device = tensor_info.device
    return [f"shape=({shape})", f"dtype={dtype}", f"device='{device}'"]

def visualize_graph(trace_result: pb.TraceResult):
    """Visualize a TraceResult's GraphModule as a list of single assignment code lines."""
    graph_module = trace_result.graph_module
    graph = graph_module.graph
    
    lines = []
    lines.append("# Graph Visualization")
    
    # Add parallel configuration information
    pc = trace_result.parallel_dims
    lines.append(f"\n# Parallel Configuration: dp_replicate={pc.dp_replicate}, dp_shard={pc.dp_shard}, " +
                f"cp={pc.cp}, tp={pc.tp}, pp={pc.pp}, world_size={pc.world_size}")
    
    # Debug info
    lines.append(f"\n# Debug info: graph has {len(graph.nodes)} nodes")

    # Add header for buffers
    lines.append("# Buffers")
    for name, info in sorted(graph_module.buffers.items(), key=lambda x: x[0]):
        line = f"buffer[{name}] = Tensor({', '.join(format_tensor_info(info))})" 
        lines.append(line)
    
    lines.append("\n# Graph Nodes (topologically sorted)")
    
    if not graph.nodes:
        lines.append("# No nodes found in the graph")
    
    # Process each node in the graph
    for node in graph.nodes:
        args_str = format_args(node.args)
        kwargs_str = format_kwargs(node.kwargs)
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        line = f"{node.name} = {node.op}[{node.target}]({params})"
        
        # Add metadata as a comment if available
        metadata = []
        if node.HasField("tensor_info"):
            info = node.tensor_info
            metadata.extend(format_tensor_info(info))

        if node.HasField('collective_meta'):
            size_str = f"{node.collective_meta.comm_tensor_size / 1e9:.2f}GB"
            ranks = sorted(node.collective_meta.group_ranks)
            n = len(node.collective_meta.group_ranks)
            stride = ranks[1] - ranks[0] if n > 1 else 0
            metadata.append(f"grp={node.collective_meta.group_desc}({size_str},{n}@{stride})")
            
        if metadata:
            line = line.ljust(120) + f"  # {', '.join(metadata)}"
            
        lines.append(line)
    
    # Mark the output node
    if hasattr(graph, 'output_node_index') and 0 <= graph.output_node_index < len(graph.nodes):
        output_node = graph.nodes[graph.output_node_index].name
        lines.append(f"\n# Graph output: {output_node}")
        
    return '\n'.join(lines)

def read_trace_result(file_path):
    """Read a TraceResult from a binary or text format file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            print(f"Read {len(content)} bytes from {file_path}")
            trace_result = pb.TraceResult.FromString(content)
    except Exception as e:
        print(f"Error parsing trace result: {e}")
        sys.exit(1)
    
    # Debug output
    print(f"Parsed TraceResult with graph module: {bool(trace_result.graph_module)}")
    if trace_result.graph_module and trace_result.graph_module.graph:
        print(f"Graph has {len(trace_result.graph_module.graph.nodes)} nodes")
            
    return trace_result

def main():
    parser = argparse.ArgumentParser(description="Visualize a TorchTitan TraceResult")
    parser.add_argument('trace_file', help='Path to the TraceResult protobuf file')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    trace_result = read_trace_result(args.trace_file)
    visualization = visualize_graph(trace_result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(visualization)
    else:
        print(visualization)

if __name__ == "__main__":
    main()
