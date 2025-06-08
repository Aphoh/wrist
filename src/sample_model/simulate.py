import argparse
from collections import Counter
import os
from math import prod
from typing import List, Iterator, Tuple
import time
from typing import Dict
import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
import torch.fx.experimental.symbolic_shapes
from torch.optim import Adam
import asyncio

from sample_model.parallel_dims import ParallelDims

from . import fx_serialize
from . import fx_graph_pb2 as pb
from .fake import FakeStore
from .model import AsyncTransformer, setup_fsdp_model



def prime_factorize(n: int) -> List[int]:
    """
    Compute the prime factorization of a number.

    Args:
        n: The number to factorize

    Returns:
        List of prime factors
    """
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n and n > 1:
            factors.append(n)
            break
    return factors


def lists_from_primes_gen(nums: List[int], d: int) -> Iterator[List[int]]:
    """
    Yield, without duplicates, every length‑d list whose ordered
    elements are products of disjoint subsets of `nums`.
    Empty subsets are allowed (slot = 1).
    """
    if d <= 0:
        raise ValueError("d must be positive")

    primes = list(Counter(nums).items())  # (prime, multiplicity)
    out = [1] * d  # current partial result

    def dfs(i: int):
        if i == len(primes):  # all primes placed
            yield out.copy()
            return

        p, c = primes[i]  # distribute c copies of p
        comp = [0] * d  # composition of c into d parts

        def comp_rec(pos: int, remaining: int):
            if pos == d - 1:  # last slot gets the rest
                comp[pos] = remaining
                for k in range(d):  # apply
                    if comp[k]:
                        out[k] *= p ** comp[k]
                yield from dfs(i + 1)
                for k in range(d):  # back‑track
                    if comp[k]:
                        out[k] //= p ** comp[k]
                return

            for cnt in range(remaining + 1):
                comp[pos] = cnt
                yield from comp_rec(pos + 1, remaining - cnt)

        yield from comp_rec(0, c)

    yield from dfs(0)


def generate_parallelism_configs(world_size: int) -> Iterator[Dict[str, int]]:
    search = ["dp_replicate", "dp_shard", "tp"]
    primes = prime_factorize(world_size)
    for assigsn in lists_from_primes_gen(primes, len(search)):
        config = dict(zip(search, assigsn))
        # Ensure that the product of all degrees equals world_size
        assert prod(config.values()) == world_size
        yield config


def parallel_config_to_str(config: Dict[str, int]) -> str:
    ws = prod(config.values())
    return f"ws{ws}" + "".join(
        f"{k}{v}" for k, v in config.items() if v > 1
    )

async def trace_model(
    *,
    vocab_size: int = 50000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 1,
    batch_size: int,
    seq_len: int = 512,
    world_size: int,
    parallel_dims: ParallelDims,
) -> Tuple[pb.TraceResult, str, Dict[str, float]]:
    """
    Trace the model and measure forward pass latency.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        batch_size: Batch size
        seq_len: Sequence length
        world_size: World size
        parallelism_config: Optional custom parallelism configuration

    Returns:
        Tuple of (TraceResult protobuf, readable graph string, timing breakdown dict)
    """
    timing = {}
    start_total = time.time()

    tp_size = parallel_dims.tp
    
    # Ensure n_heads is divisible by tp_size
    if n_heads % tp_size != 0:
        n_heads = (n_heads // tp_size) * tp_size
        if n_heads == 0:
            n_heads = tp_size

    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        static_shapes=True, allow_non_fake_inputs=True, shape_env=shape_env
    )
    device_mesh = parallel_dims.build_mesh()
    
    device = torch.device("cuda")
    with fake_mode, device:
        start_model = time.time()
        tp_mesh = parallel_dims.get_tp_mesh(device_mesh)
        model = AsyncTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tp_group=tp_mesh.get_group() if tp_mesh is not None else None, 
        )
                                
        model = setup_fsdp_model(model, parallel_dims, device_mesh=device_mesh)
        timing["model_init"] = time.time() - start_model

        # Create mock optimizers
        optimizers = Adam(model.parameters(), lr=0.001)

        # Create input tensors
        adjusted_batch_size = batch_size // (parallel_dims.dp_shard * parallel_dims.dp_replicate)
        inputs = torch.randint(0, vocab_size, (adjusted_batch_size, seq_len), dtype=torch.int32)
        labels = inputs.clone().detach().to(torch.long)

        def step():
            optimizers.zero_grad()

            # Forward pass
            pred = model(inputs)
            
            # Simple loss computation - account for tensor parallel output size
            loss = torch.nn.functional.cross_entropy(
                pred.view(-1, pred.size(-1)), 
                labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Mock gradient clipping
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            optimizers.step()
            return optimizers.state_dict()

        start_tracing = time.time()
        make_fx_out = torch.fx.experimental.proxy_tensor.make_fx(
            step,
            tracing_mode="fake",
            decomposition_table={},
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )()
        
        # Clean up graph
        for node in make_fx_out.graph.nodes:
            if node.op == "placeholder" and "val" not in node.meta:
                make_fx_out.graph.erase_node(node)
        make_fx_out.graph.eliminate_dead_code()
        make_fx_out.recompile()
        timing["tracing"] = time.time() - start_tracing

        # Add tensor info to nodes that have fake tensors
        for node in make_fx_out.graph.nodes:
            meta_val = node.meta.get('val')
            if meta_val is None:
                meta_val = node.meta.get('example_value')
            if isinstance(meta_val, torch._subclasses.FakeTensor):
                if 'val' not in node.meta:
                    node.meta['val'] = meta_val

        start_export = time.time()
        # fx_serialize.serialize now returns a protoc GraphModuleData
        graph_module_proto = fx_serialize.serialize(make_fx_out)
        timing["export"] = time.time() - start_export

        timing["total"] = time.time() - start_total

        # Create ParallelConfig protobuf message
        parallel_config_proto = pb.ParallelConfig(
            dp_replicate=parallel_dims.dp_replicate,
            dp_shard=parallel_dims.dp_shard,
            cp=1,
            tp=parallel_dims.tp,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )

        # Create TraceResult protobuf message
        result_proto = pb.TraceResult(
            parallel_dims=parallel_config_proto,
            graph_module=graph_module_proto
        )

        return result_proto, make_fx_out.print_readable(), timing


async def main() -> None:
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Simulate model tracing and parallelism")
    parser.add_argument("--world_size", type=int, default=8, help="World size for distributed training")
    parser.add_argument("--rank", type=int, default=0, help="World size for distributed training")
    args = parser.parse_args()

    rank: int = args.rank
    world_size: int = args.world_size
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    setup_start = time.time()
    torch.distributed.init_process_group(
        backend="fake",
        world_size=world_size,
        store=FakeStore(),
        rank=rank,
    )
    setup_time = time.time() - setup_start
    print(f"Setup time: {setup_time:.4f} seconds")

    print(f"World size: {world_size}")
    print(f"Prime factorization: {prime_factorize(world_size)}")

    # Model configuration
    vocab_size = 50000
    d_model = 512
    n_heads = 8
    n_layers = 8
    seq_len = 512
    
    # Calculate batch size (4M tokens total)
    total_tokens = 4 * 1024 * 1024
    batch_size = total_tokens // seq_len
    assert batch_size > 0, "Batch size must be greater than 0"
    
    print(f"Total tokens: {total_tokens}, Seq length: {seq_len}, Batch size: {batch_size}")

    results = []

    # Generate all configurations in advance to show progress
    configs = list(generate_parallelism_configs(world_size))
    configs = [c for c in configs if (c["tp"] <= n_heads) and (c["dp_shard"]*c["dp_replicate"] <= batch_size)]
    total_configs = len(configs)

    # Try all possible parallelism configurations
    async def run_config(i: int, config: Dict[str, int]):
        print(f"Testing configuration {i + 1}/{total_configs}: {config}")
        # trace_model now returns a protoc TraceResult
        parallel_dims = ParallelDims(world_size=world_size, **config)
        proto, code, timing = await trace_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            world_size=world_size,
            parallel_dims=parallel_dims,
        )
        # Serialize the protobuf message to bytes
        out_bytes = proto.SerializeToString()
        name = parallel_config_to_str(config)
        # Write binary protobuf data
        with open(f"traces/{name}.pb", "wb") as f:
            f.write(out_bytes)
        # Write readable graph code
        with open(f"traces/{name}.py", "w") as f:
            f.write("# Generated code for parallel config: " + name + "\n")
            f.write("import torch\n")
            f.write(code)
        return f"Config {i}", timing

    os.makedirs("traces", exist_ok=True)
    # Run all configurations
    for i, config in enumerate(configs):
        result = await run_config(i, config)
        results.append(result)

    # Print summary of results
    print("\n===== RESULTS SUMMARY =====")
    results.sort(key=lambda x: x[1]["total"])  # Sort by latency
    for config_name, timing in results:
        timing_str = ' '.join([f'{k}: {v:.4f}' for k, v in timing.items()])
        print(f"{config_name}: {timing_str}")


if __name__ == "__main__":
    asyncio.run(main())
