import datetime
import torch
import torch.distributed as dist
import argparse
import csv
import os
import math


def _repeated_subgroup(group_size, repeats, base=0):
    """
    Outputs the sequence
    [base, base+1, base+2, ..., base+group_size-1]
    repeated `repeats` times
    """
    repeated = [base + i for i in range(group_size)]
    return [a for _ in range(repeats) for a in repeated]


def strided_indices(world_size, stride, group_size):
    num_groups = world_size // group_size
    group_idxs = []
    strided_subgroup_size = group_size * stride
    n_repeats = strided_subgroup_size // stride
    for base in range(0, num_groups, stride):
        subgroup = _repeated_subgroup(stride, n_repeats, base)
        group_idxs.extend(subgroup)
    # Map group_idxs back to ranks
    groups = [[] for _ in range(num_groups)]
    for i, idx in enumerate(group_idxs):
        groups[idx].append(i)
    return groups


def get_groups(world_size, rank, ranks_per_node):
    assert math.log2(
        ranks_per_node
    ).is_integer(), "ranks_per_node should be a power of 2"
    assert (
        world_size % ranks_per_node == 0
    ), "world_size should be divisible by ranks_per_node"
    group_n_gpus = 2
    stride = 1
    groups = []
    while group_n_gpus <= world_size:
        num_groups = world_size // group_n_gpus
        if rank == 0:
            print(f"num_gpus{group_n_gpus}")
        while stride <= num_groups:
            stride_groups = strided_indices(world_size, stride, group_n_gpus)
            if rank == 0:
                print(f"    stride{stride:02}: {stride_groups}")
            for stride_group in stride_groups:
                group = dist.new_group(stride_group, backend="nccl")
                if rank in stride_group:
                    groups.append((group_n_gpus, stride, stride_group, group))
            stride *= 2
        stride = 1
        group_n_gpus *= 2
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples per test"
    )
    parser.add_argument(
        "--output", type=str, default="output.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    print(
        f"world_size: {world_size}, rank: {rank}, local_world_size: {local_world_size}, local_rank: {local_rank}, device: {torch.cuda.current_device()}",
        flush=True,
    )
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=30)
    )
    dist.barrier()

    hostname = os.environ["HOSTNAME"]
    hostnames = [None for _ in range(world_size)]
    if rank == 0:
        print("Getting hostnames")
    dist.all_gather_object(hostnames, hostname)
    if rank == 0:
        for i, h in enumerate(hostnames):
            print(f"Rank {i} is on {h}")
    assert (
        local_rank == rank % local_world_size
    ), f"local_rank {local_rank} should be equal to rank {rank} % local_world_size {local_world_size}"

    groups = get_groups(world_size, rank, local_world_size)
    # List of data sizes in MB
    data_sizes_mb = [2 * (2**i) for i in range(11)]
    operations = ["reduce_scatter", "all_reduce", "ring", "all_gather"]

    if rank == 0:
        csv_file = open(args.output, "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "operation",
                "data_size_mb",
                "num_gpus",
                "stride",
                "sample_idx",
                "duration_sec",
            ]
        )

    MAX_MEMORY_PER_PROCESS = 20 * 1024 * 1024 * 1024 # 20gb

    for op in operations:
        for data_size_mb in data_sizes_mb:
            data_size = data_size_mb * 1024 * 1024 # convert mb to bytes
            for num_gpus, stride, group_ranks, group in groups:
                group_size = len(group_ranks)
                if op in ["all_reduce", "reduce_scatter"]:
                    memory_per_process = data_size
                elif op == "all_gather" or op == "gather":
                    memory_per_process = data_size * group_size
                elif op == "ring":
                    memory_per_process = 2 * data_size
                else:
                    raise ValueError(f"Unknown operation {op}")
                if memory_per_process > MAX_MEMORY_PER_PROCESS:
                    if rank == 0:
                        print(
                            f"Skipping {op} with data size {data_size_mb}MB for group {num_gpus} due to memory limit",
                            flush=True,
                        )
                    continue
                # Create tensor of appropriate size
                num_elements = data_size // 2  # float16, 2 bytes per element
                tensor = torch.rand(num_elements, device="cuda", dtype=torch.float16)
                for sample in range(args.samples):
                    if rank == 0:
                        print(
                            f"N{num_gpus:02}, S{stride:02}: Data: {data_size_mb:04}MB Operation {op}",
                            flush=True,
                        )
                    dist.barrier()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    if op == "all_reduce":
                        dist.all_reduce(tensor, group=group)
                    elif op == "reduce_scatter":
                        output = torch.zeros(
                            num_elements // group_size,
                            device="cuda",
                            dtype=torch.float16,
                        )
                        dist.reduce_scatter_tensor(output, tensor, group=group)
                    elif op == "all_gather":
                        tensors = [torch.zeros_like(tensor) for _ in group_ranks]
                        dist.all_gather(tensors, tensor, group=group)
                    elif op == "ring":
                        my_ind = group_ranks.index(rank)
                        next_ind = (my_ind + 1) % len(group_ranks)
                        prev_ind = (my_ind - 1) % len(group_ranks)
                        next_rank = group_ranks[next_ind]
                        prev_rank = group_ranks[prev_ind]
                        other_tensor = torch.zeros_like(
                            tensor, device=torch.cuda.current_device()
                        )
                        send = dist.P2POp(dist.isend, tensor, next_rank, group=group)
                        recv = dist.P2POp(
                            dist.irecv, other_tensor, prev_rank, group=group
                        )
                        res = dist.batch_isend_irecv([send, recv])
                        for r in res:
                            r.wait()
                        dist.barrier(
                            group=group
                        )  # I think we need another barrier here, since rank 0 could just be special

                    else:
                        raise ValueError(f"Unknown operation {op}")
                    end_event.record()
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000.0  # seconds

                    avg_duration = torch.tensor(duration).to(device='cuda')
                    dist.all_reduce(avg_duration, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize()
                    avg_duration = (avg_duration / world_size).cpu().item()

                    if rank == 0:
                        csv_writer.writerow(
                            [op, data_size_mb, num_gpus, stride, sample, avg_duration]
                        )
                        csv_file.flush()
                        print(
                            f"N{num_gpus:02}, S{stride:02}: Data: {data_size_mb:04}MB Operation {op}, completed in {1000 * avg_duration:.2f} ms",
                            flush=True,
                        )
                    torch.cuda.empty_cache()

    if rank == 0:
        csv_file.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
