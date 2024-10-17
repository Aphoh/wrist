import datetime
import torch
import torch.distributed as dist
import argparse
import time
import csv
import os
import math

def get_groups(world_size, rank, ranks_per_node):
    assert math.log2(ranks_per_node).is_integer(), "ranks_per_node should be a power of 2"
    assert world_size % ranks_per_node == 0, "world_size should be divisible by ranks_per_node"
    group_n_gpus = 2
    
    groups = []
    tgroups = []
    while group_n_gpus <= world_size:
        for group_rank_0 in range(0, world_size, group_n_gpus):
            group_ranks = list(range(group_rank_0, group_rank_0 + group_n_gpus))
            group = dist.new_group(group_ranks, backend='nccl')
            if rank in group_ranks:
                print(f"ngpu {group_n_gpus} rank {rank} is in group {group_ranks}")
                groups.append((group_n_gpus, group_ranks, group))
        nt = world_size // group_n_gpus
        for tgroup_rank_0 in range(0, nt):
            tgroup_ranks = list(range(tgroup_rank_0, world_size, nt))
            group = dist.new_group(tgroup_ranks, backend='nccl')
            if rank in tgroup_ranks:
                print(f"ngpu {group_n_gpus} rank {rank} is in tgroup {tgroup_ranks}")
                tgroups.append((group_n_gpus, tgroup_ranks, group))

        group_n_gpus *= 2
    return groups, tgroups

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10, help='Number of samples per test')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file')
    args = parser.parse_args()


    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])


    torch.cuda.set_device(local_rank)
    print(f"world_size: {world_size}, rank: {rank}, local_world_size: {local_world_size}, local_rank: {local_rank}, device: {torch.cuda.current_device()}", flush=True)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=30))
    dist.barrier()
    hostname = os.environ["HOSTNAME"]
    hostnames = [None for _ in range(world_size)]
    dist.all_gather_object(hostnames, hostname)
    if rank == 0:
        for i, h in enumerate(hostnames):
            print(f"Rank {i} is on {h}") 
    assert local_rank == rank % local_world_size, f"local_rank {local_rank} should be equal to rank {rank} % local_world_size {local_world_size}"

    seq_groups, transpose_groups = get_groups(world_size, rank, local_world_size)
    # List of data sizes in MB
    data_sizes_mb = [2 * (2 ** i) for i in range(12)]  # 2MB to 4096MB

    operations = ['all_reduce', 'ring', 'reduce', 'all_gather', 'gather']

    # Open CSV file per rank
    if rank == 0:
        csv_file = open(args.output, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['operation', 'data_size_mb', 'group_name', 'transpose', 'sample', 'duration_sec'])

    MAX_MEMORY_PER_PROCESS = 20 * 1024 * 1024 * 1024  # 20GB

    for op in operations:
        for data_size_mb in data_sizes_mb:
            data_size = data_size_mb * 1024 * 1024  # Convert MB to bytes
            for transpose in [False, True]:
                groups = transpose_groups if transpose else seq_groups
                for group_name, group_ranks, group in groups:
                    if rank in group_ranks:
                        group_size = len(group_ranks)
                        # Estimate memory required per process
                        if op in ['all_reduce', 'reduce']:
                            memory_per_process = data_size
                        elif op == 'all_gather' or op == 'gather':
                            memory_per_process = data_size * group_size
                        elif op == 'ring':
                            memory_per_process = 2 * data_size
                        else:
                            raise ValueError(f'Unknown operation {op}')
                        if memory_per_process > MAX_MEMORY_PER_PROCESS:
                            if rank == 0:
                                print(f"Skipping {op} with data size {data_size_mb}MB for group {group_name} due to memory limit", flush=True)
                            continue
                        # Create tensor of appropriate size
                        num_elements = data_size // 2  # float16, 2 bytes per element
                        tensor = torch.rand(num_elements, device='cuda', dtype=torch.float16)
                        for sample in range(args.samples):
                            if rank == 0:
                                print(f"{group_name}: Running {op} {'T' if transpose else 'N' } with data size {data_size_mb}MB", flush=True)
                            dist.barrier()
                            torch.cuda.synchronize()
                            start_time = time.time()
                            if op == 'all_reduce':
                                dist.all_reduce(tensor, group=group)
                            elif op == 'reduce':
                                dist.reduce(tensor, dst=group_ranks[0], group=group)
                            elif op == 'all_gather':
                                tensors = [torch.zeros_like(tensor) for _ in group_ranks]
                                dist.all_gather(tensors, tensor, group=group)
                            elif op == 'gather':
                                if rank == group_ranks[0]:
                                    tensors = [torch.zeros_like(tensor) for _ in group_ranks]
                                    dist.gather(tensor, gather_list=tensors, dst=group_ranks[0], group=group)
                                else:
                                    dist.gather(tensor, dst=group_ranks[0], group=group)
                            elif op == 'ring':
                                my_ind = group_ranks.index(rank)
                                next_ind = (my_ind + 1) % len(group_ranks)
                                prev_ind = (my_ind - 1) % len(group_ranks)
                                next_rank = group_ranks[next_ind]
                                prev_rank = group_ranks[prev_ind]
                                other_tensor = torch.zeros_like(tensor, device=torch.cuda.current_device())
                                send = dist.P2POp(dist.isend, tensor, next_rank, group=group)
                                recv = dist.P2POp(dist.irecv, other_tensor, prev_rank, group=group)
                                res = dist.batch_isend_irecv([send, recv])
                                for r in res:
                                    r.wait()
                                dist.barrier(group=group) # I think we need another barrier here, since rank 0 could just be special

                            else:
                                raise ValueError(f'Unknown operation {op}')
                            torch.cuda.synchronize()
                            duration = time.time() - start_time

                            # Write to CSV
                            if rank == 0:
                                csv_writer.writerow([op, data_size_mb, group_name, transpose, sample, duration])
                                csv_file.flush()
                                print(f"Completed {op} with data size {data_size_mb}MB for ngpu={group_name} completed in {1000 * duration:.2f} ms", flush=True)
                            torch.cuda.empty_cache()

    if rank == 0:
        csv_file.close()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
