import torch
import torch.export
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._inductor.lowering import lowerings
from torch._inductor.utils import fresh_inductor_cache

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        
    def forward(self, x):
        # Simple operations that will create interesting IR
        y = self.linear(x)
        z = torch.relu(y)
        return torch.sum(z, dim=1)

def export_and_lower_model():
    # Create model and example inputs
    model = SimpleModel()
    example_inputs = (torch.randn(3, 10),)
    
    # Export the model to get a GraphModule
    print("=== Exporting model with torch.export ===")
    exported_program = torch.export.export(model, example_inputs, strict=True)
    gm = exported_program.module()
    
    print(f"Exported GraphModule type: {type(gm)}")
    print(f"Number of FX nodes: {len(list(gm.graph.nodes))}")
    print("\nFX Graph:")
    print(gm.graph)
    
    # Use fresh_inductor_cache to avoid registry conflicts
    with fresh_inductor_cache():
        # Now use GraphLowering to get optimized IR
        print("\n=== Lowering to Inductor IR ===")
        graph_lowering = GraphLowering(
            gm=gm,
            example_inputs=example_inputs,
            is_inference=True,
            layout_opt=True,
        )
        
        # IMPORTANT: Set up the virtualized context before running
        with V.set_graph_handler(graph_lowering):
            # Run the lowering process
            result = graph_lowering.run(*example_inputs)
    
    # Access the optimized IR
    print(f"\nInductor IR Analysis:")
    print(f"Number of output nodes: {len(graph_lowering.graph_outputs)}")
    print(f"Number of buffers: {len(graph_lowering.buffers)}")
    print(f"Number of operations: {len(graph_lowering.operations)}")
    print(f"Device types: {graph_lowering.device_types}")
    
    # Print details about each output
    print(f"\n=== Graph Outputs ===")
    for i, output in enumerate(graph_lowering.graph_outputs):
        print(f"Output {i}: {type(output)} - {output}")
        if hasattr(output, 'get_size'):
            print(f"  Size: {output.get_size()}")
        if hasattr(output, 'get_dtype'):
            print(f"  Dtype: {output.get_dtype()}")
    
    # Print details about buffers (intermediate results)
    print(f"\n=== Buffers ===")
    for i, buffer in enumerate(graph_lowering.buffers):
        print(f"Buffer {i}: {buffer.get_name()} - {type(buffer)}")
        if hasattr(buffer, 'get_size'):
            print(f"  Size: {buffer.get_size()}")
        if hasattr(buffer, 'get_dtype'):
            print(f"  Dtype: {buffer.get_dtype()}")
    
    # Print details about operations
    print(f"\n=== Operations ===")
    for i, op in enumerate(graph_lowering.operations):
        print(f"Operation {i}: {op.get_name()} - {type(op)}")
        if hasattr(op, 'get_reads'):
            reads = [dep.name for dep in op.get_reads()]
            print(f"  Reads: {reads}")
    
    # Print graph inputs
    print(f"\n=== Graph Inputs ===")
    for name, input_node in graph_lowering.graph_inputs.items():
        print(f"Input '{name}': {type(input_node)}")
        if hasattr(input_node, 'get_size'):
            print(f"  Size: {input_node.get_size()}")
    
    return graph_lowering

def analyze_specific_ir_types(graph_lowering):
    """Analyze specific types of IR nodes"""
    from torch._inductor.ir import Pointwise, Reduction, ComputedBuffer, InputBuffer
    
    print("\n=== IR Node Type Analysis ===")
    
    # Count different types
    pointwise_count = 0
    reduction_count = 0
    computed_buffer_count = 0
    input_buffer_count = 0
    
    all_nodes = list(graph_lowering.buffers) + list(graph_lowering.operations)
    
    for node in all_nodes:
        if isinstance(node, Pointwise):
            pointwise_count += 1
            print(f"Pointwise: {node.get_name()}")
        elif isinstance(node, Reduction):
            reduction_count += 1
            print(f"Reduction: {node.get_name()}")
        elif isinstance(node, ComputedBuffer):
            computed_buffer_count += 1
            print(f"ComputedBuffer: {node.get_name()}")
        elif isinstance(node, InputBuffer):
            input_buffer_count += 1
            print(f"InputBuffer: {node.get_name()}")
    
    print(f"\nSummary:")
    print(f"  Pointwise ops: {pointwise_count}")
    print(f"  Reduction ops: {reduction_count}")
    print(f"  Computed buffers: {computed_buffer_count}")
    print(f"  Input buffers: {input_buffer_count}")

def main():
    # Export and lower the model
    graph_lowering = export_and_lower_model()
    
    # Analyze specific IR types
    analyze_specific_ir_types(graph_lowering)
    
    # Optional: If you want to see the generated code
    print("\n=== Generated Code Preview ===")
    try:
        # Set up context for code generation
        with V.set_graph_handler(graph_lowering):
            # Initialize wrapper code generation
            graph_lowering.init_wrapper_code()
            
            # Generate the actual optimized code
            wrapper_code, kernel_code = graph_lowering.codegen()
        
        if hasattr(wrapper_code, 'value'):
            code_lines = wrapper_code.value.split('\n')
            print("First 20 lines of generated wrapper code:")
            for i, line in enumerate(code_lines[:20]):
                print(f"{i+1:3d}: {line}")
            if len(code_lines) > 20:
                print(f"... (truncated, total {len(code_lines)} lines)")
    except Exception as e:
        print(f"Code generation failed: {e}")

if __name__ == "__main__":
    main()