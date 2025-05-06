# Neural Network Simulation on torch.fx Graphs

Efficiently training these large models with long context capabilities requires using a combination of many parallelism techniques (data, pipeline, contex, and tensor parallelism) but finding optimal configurations remains difficult.

Additionally, current auto-parallelism solutions have several drawbacks:
- They lack flexibility to accommodate custom parallelization strategies like context parallelism
- They often require rewriting code in specific languages or frameworks
- They struggle to incorporate new forms of parallelism
- They require expensive cluster downtime for profiling and searching for optimal configurations

The goal of this project is to solve these issues by using already developed parallelism strategies and densely searching over all possible configurations using simulation.

## Basic idea

In order to estimate the execution time of the network, we can build a simple regression model for each of collective operations we care about (allreduce, allgather, reduce scatter, etc). Importantly, this only needs to be done **once per compute cluster**, and can be reused for all models.

For each model, there are three stages:
1. **Tracing**: We use `torch.fx.experimental.proxy_tensor` to trace model execution on cpu and generate an intermediate representation of the model's forward/backward pass, along with all collective operations. This must be done once for each assignment of `DP=...,TP=..., etc` which scales roughly as $\mathcal{O}(\log_2(N)^{t-1})$ for $N$ accelerators and $t$ parallelism strategies. For 1024 accelerators and 5 parallelism strategies, this is around 1,000 traces, many of which can be filtered out as they are impossible due to memory constraints. 
Each trace is exported into [our own format](../src/graph/torch_titan.proto) that's read in by [the simulation code](../src/bin/sim_fx.rs).
2. **Profiling**: We extract each operation, argument shapes, strides, dtype, etc and profile them on a single accelerator in order to estimate the execution time of each operation. These can be cached for future use, or could be stored in a remote database. There is also a significant amount of overlap between the operations in each trace.
3. **Simulation**: We build an execution graph from each traced model, assuming all CUDA operations are executed sequentially and collectives are potentially parallelized[^1].  Additionally, we can search over every possible ordering of the network. See [this](https://github.com/NVIDIA/Megatron-LM/blob/10b5c5898e950a4ecc62974ed834163713868921/megatron/core/parallel_state.py#L572) for more information.

[^1]: This depends on whether `CUDA_DEVICE_MAX_CONNECTIONS=1` is set, as is the default in Megatron, but for now it would be fair to assume that a single collective over NVLink and a single collective over infiniband can be run concurrently.

## Scale of the Problem: Combinatorial Explosion

The search space for parallelism strategies grows rapidly with cluster size. If we have a cluster of $N=2^n$ nodes and $k$ total parallelism strategies $s_1, s_2, \ldots, s_k$, such that $n = \sum_i^k n_{s_i}$, we sum over the number of in-use parallelism strategies (for which $n_{s_i} \neq 0$).

For each case, we should choose $t$ of the $k$ total strategies, giving a factor of $\binom{k}{t}$. There are then $t!$ ways to order the strategies in the network, and $\binom{n - t + t - 1}{t - 1} = \binom{n - 1}{t - 1}$ ways to assign non-zero values to the chosen $s_i$ (a stars and bars problem).

This yields a total number of searchable partitioning strategies at around:

$$\sum_{t=1}^k \binom{k}{t} t! \binom{n-1}{t-1}$$

For a cluster with $2^{14} = 16,384$ accelerators and 6 different parallelism strategies, this yields approximately 1.55 million possible combinations. The majority of these strategies may be impossible due to memory constraints, but the search space remains enormous.
Our method solves this by building ~3000 traces, compute graphs for each, then iterating over each network permutation and searching for the best configuration.
Each individual simulation can be done in a few milliseconds.

## Using FakeTensor and proxy_tensor.make_fx for Graph Generation

To generate a compute graph from arbitrary PyTorch models:

1. We can use `FakeTensor` to run the model without actual computation, allowing us to extract the operations
   - `FakeTensor` simulates tensor operations without allocating memory
   - This provides shape and dtype information without running actual computations

2. `proxy_tensor.make_fx` enables us to generate a graph representation of the model's execution
   - This captures the entire computational flow including operations, dependencies, and shapes
   - The resulting graph can be analyzed to identify high-intensity operations

The output of this graph might have operations that look like this fsdp allgather efore a forward operation:
```python
def step(..):
    ...
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([getitem_24, getitem_25, getitem_26], [8208384, 512, 8208384], 16417280, 32, 0, torch.bfloat16, device(type='cuda', index=0));  getitem_24 = getitem_25 = getitem_26 = None
    getitem_27: "bf16[16417280]" = all_gather_copy_in[0]
    getitem_28: "bf16[525352960]" = all_gather_copy_in[1];  all_gather_copy_in = None
    _torchbind_obj0 = self._torchbind_obj0
    _allgather_base_ = torch.ops.c10d._allgather_base_.default(getitem_28, getitem_27, _torchbind_obj0, False);  getitem_28 = getitem_27 = _torchbind_obj0 = None
```

## Current Limitations

* Tracing through a model with a single layer takes ~1-2 seconds, but as there's no `Scan`-like primitive in PyTorch, tracing many layers scales linearly. 
Hence scannning longer models would require specially tracing a single layer and then using that to build the longer model.

* Pipeline parallelism currently isn't supported, as it requires running traces for each stage and for each microbatch within a given batch. This would involve considerably more work, but shouldn't be impossible.

* `torch.compile` is not supported by `proxy_tensor`, so it's difficult to model what operations could be fused together. This would almost certainly be solveable using `torch.export` or a little more effort on tracing. For now we can probably just assume timing will be dominated by matmuls/attention/collectives, and ignore the rest.
