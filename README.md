# WIP - Wrist

Wrist is a currently VERY WIP simulation tool for analyzing and optimizing PyTorch models across different distributed training configurations by simulating their execution with various sharding strategies.

## Overview

Large neural network models require a combination of parallelism techniques (data, tensor, pipeline, context parallelism) to train efficiently. Finding the optimal configuration for these techniques is challenging and typically requires expensive trial and error on real hardware.

Wrist solves this problem by simulating the execution of PyTorch models with different parallelism configurations, allowing you to find optimal strategies without consuming valuable cluster time.

For more information on the motivation and technical details behind this approach, see [MOTIVATION.md](./docs/MOTIVATION.md).

## How It Works

Wrist uses a three-stage approach:

1. **Tracing**: Generate serialized FX graphs from PyTorch models using different parallelism configurations
2. **Profiling**: Extract operations, shapes, and datatypes to estimate execution time
3. **Simulation**: Build and analyze compute graphs to find optimal configurations

## Usage

### Generating Traces

Traces are generated using the [`torchtitan`](https://github.com/Aphoh/torchtitan) library. The simulation code is available at:
[https://github.com/Aphoh/torchtitan/blob/main/torchtitan/simulate.py](https://github.com/Aphoh/torchtitan/blob/main/torchtitan/simulate.py)

### Running Simulations

The `sim_fx` binary analyzes trace files to find the optimal configuration:

```
# Example: Simulate with traces directory and 80GB memory per GPU
$ cargo run --bin sim_fx -- -t ./traces -l 80.0

Runtime distribution:
[155.0 ms, 193.8 ms): 37 #####################################
[193.8 ms, 232.5 ms): 9  #########
[232.5 ms, 271.3 ms): 5  #####
[271.3 ms, 310.0 ms): 3  ###
[310.0 ms, 348.8 ms): 0  
[348.8 ms, 387.5 ms): 0  
[387.5 ms, 426.3 ms): 1  #
[426.3 ms, 465.0 ms): 0  
[465.0 ms, 503.8 ms): 0  
[503.8 ms, 542.5 ms): 3  ###

Best trace: PC{ dp_shard: 2, cp: 2, tp: 2, pp: 1 }
Time: 155032 us
```

## Key Features

- **Multiple Parallelism Dimensions**:
  - Data Parallelism (DP) - both sharded and replicated
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
  - Context Parallelism (CP)

- **Comprehensive Analysis**:
  - Communication costs (AllReduce, AllGather, ReduceScatter, AllToAll)
  - Memory requirements (parameters and activations)
  - Computation time for key operations (MatMul, Flash Attention, etc.)

- **Memory-Aware Simulation**:
  - Filters out configurations that would exceed available GPU memory
  - Provides memory usage estimates for parameters

## Current Limitations
- Tracing through multi-layer models scales linearly with the number of layers
- Pipeline parallelism is not yet supported
- Does not yet accurately account for `torch.compile` optimizations