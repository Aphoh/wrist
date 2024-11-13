use crate::network::{Collective, CollectiveType};
use crate::ops::{MemoryProfile, Operation};
use crate::sharding::SeqModelSpec;
use crate::sharding::ShardStrategy;
use crate::sharding::ShardingType;

use super::{ComputeUnit, Kernel}; // Add this line to import the Operation trait

pub struct MLP {
    pub input_size: u64,
    pub intermediate_size: u64,
    pub output_size: u64,
}

impl Operation for MLP {
    fn forward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        downstream_collective: Option<Collective>,
    ) -> (Vec<ComputeUnit>, Option<Collective>) {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let leaf_intermediate_size = self.intermediate_size / feature;
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;
        let w1_bytes = 2 * self.input_size * leaf_intermediate_size;
        let w2_bytes = 2 * self.output_size * leaf_intermediate_size;

        let output_act_bytes = 2 * leaf_batch_size * leaf_seq_size * self.output_size;

        let compute = vec![
            ComputeUnit::new(
                vec![Kernel::MatMul {
                    m: leaf_batch_size * leaf_seq_size,
                    n: self.intermediate_size,
                    k: self.input_size,
                }],
                strategy.collective(
                    // All-gather w2
                    ShardingType::Tensor,
                    CollectiveType::AllGather,
                    w2_bytes,
                ),
            ),
            ComputeUnit::new(
                vec![Kernel::MatMul {
                    m: leaf_batch_size * leaf_seq_size,
                    n: self.output_size,
                    k: leaf_intermediate_size,
                }],
                downstream_collective,
            ),
            ComputeUnit::new(
                vec![],
                strategy.collective(
                    ShardingType::Tensor,
                    CollectiveType::AllReduce,
                    output_act_bytes,
                ),
            ),
        ];
        // all-gather the w1 gradients before we start
        let downstream_collective =
            strategy.collective(ShardingType::Data, CollectiveType::AllGather, w1_bytes / batch);
        (compute, downstream_collective)
    }

    fn backward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        upstream_collective: Option<Collective>,
    ) -> (Vec<ComputeUnit>, Option<Collective>) {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let leaf_intermediate_size = self.intermediate_size / feature;
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;

        let input_act_bytes = 2 * leaf_batch_size * leaf_seq_size * self.input_size;
        let w2_bytes = 2 * self.output_size * leaf_intermediate_size;

        // input -> w1 -> intermediate -> w2 -> output
        let compute = vec![
            // Upstream gradients are size [B*S, O]
            // Gradient of intermediate activations (dl/doutput w2^T)
            ComputeUnit {
                kernels: vec![Kernel::MatMul {
                    m: leaf_batch_size * leaf_seq_size,
                    k: self.output_size,
                    n: leaf_intermediate_size,
                }],
                collective: upstream_collective, // This is the reduce-scatter of the upstream gradients
            },
            // Gradient of w2 (intermediate^T dl/doutput)
            ComputeUnit {
                kernels: vec![
                    Kernel::MatMul {
                        // Compute the gradient of the intermediate activations
                        m: leaf_intermediate_size,
                        k: leaf_batch_size * leaf_seq_size,
                        n: self.output_size,
                    }, // TODO: add in the kernel for the activation backprop
                ],
                collective: None,
            },
            // Gradients are of size [B*S, I]
            // Gradient of intermediate activations (dl/dintermediate w1^T)
            ComputeUnit {
                kernels: vec![Kernel::MatMul {
                    m: leaf_batch_size * leaf_seq_size,
                    k: leaf_intermediate_size,
                    n: self.input_size,
                }],
                collective: strategy.collective(
                    // reduce-scatter gradients
                    ShardingType::Data,
                    CollectiveType::ReduceScatter,
                    w2_bytes,
                ),
            },
            // Gradient of w1 (input^T dl/dintermediate)
            ComputeUnit {
                kernels: vec![Kernel::MatMul {
                    // Compute the gradient of the intermediate activations
                    m: self.input_size,
                    k: leaf_batch_size * leaf_seq_size,
                    n: leaf_intermediate_size,
                }],
                collective: strategy.collective(
                    // tp all-reduce input gradients
                    ShardingType::Tensor,
                    CollectiveType::AllReduce,
                    input_act_bytes,
                ),
            }, // TODO: layernorm
        ];
        // reduce-scatter the w1 gradients in earlier layers
        let downstream_collective = strategy.collective(
            ShardingType::Tensor,
            CollectiveType::ReduceScatter,
            2 * self.input_size * leaf_intermediate_size,
        );
        (compute, downstream_collective)
    }

    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> MemoryProfile {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let leaf_hidden_size = self.intermediate_size / feature;
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;

        let w1_memory = 2 * self.input_size * leaf_hidden_size;
        let w2_memory = 2 * leaf_hidden_size * self.output_size;
        let w1_act_memory = 2 * leaf_batch_size * leaf_seq_size * self.input_size;
        let w2_act_memory = 2 * leaf_batch_size * leaf_seq_size * leaf_hidden_size;
        let weight_memory = w1_memory + w2_memory;
        let activation_memory = 2 * leaf_batch_size * leaf_seq_size * self.output_size;

        MemoryProfile {
            weight_memory,
            activation_memory,
            cache_for_backprop: w1_act_memory + w2_act_memory,
            gradient_size: weight_memory,
        }
    }

    fn validate(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> bool {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();
        let batch_per_leaf = axes.batch / batch;
        let sequence_per_leaf = axes.sequence / sequence;
        let output_per_leaf = self.output_size / feature;
        batch_per_leaf != 0 && sequence_per_leaf != 0 && output_per_leaf != 0
    }
}
