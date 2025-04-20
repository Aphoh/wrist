use crate::kernels::Kernel;
use crate::network::{Collective, CollectiveType};
use crate::ops::{MemoryProfile, Operation};
use crate::sharding::SeqModelSpec;
use crate::sharding::ShardStrategy;
use crate::sharding::ShardingType;
use crate::utils::ValidationError;

use super::ComputeUnit;

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
            ComputeUnit::single(
                Kernel::matmul(
                    "input * w1 matmul",
                    leaf_batch_size * leaf_seq_size,
                    self.input_size,
                    leaf_intermediate_size,
                ),
                strategy.collective(
                    "All-gather w2",
                    ShardingType::Data,
                    CollectiveType::AllGather,
                    w2_bytes / batch,
                ),
            ),
            ComputeUnit::single(
                Kernel::matmul(
                    "intermediate * w2 matmul",
                    leaf_batch_size * leaf_seq_size,
                    leaf_intermediate_size,
                    self.output_size,
                ),
                downstream_collective,
            ),
            ComputeUnit::conly(strategy.collective(
                "AllReduce mlp output",
                ShardingType::Tensor,
                CollectiveType::AllReduce,
                output_act_bytes,
            )),
        ];

        let downstream_collective = strategy.collective(
            "All-gather w1",
            ShardingType::Data,
            CollectiveType::AllGather,
            w1_bytes / batch,
        );
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
            ComputeUnit::single(
                Kernel::matmul(
                    "intermediate act gradients: upstream * w2^T",
                    leaf_batch_size * leaf_seq_size,
                    self.output_size,
                    leaf_intermediate_size,
                ),
                upstream_collective, // This is the reduce-scatter of the upstream gradients
            ),
            // Gradient of w2 (intermediate^T dl/doutput)
            ComputeUnit::konly(Kernel::matmul(
                "w2 gradient: intermediate^T * upstream",
                leaf_intermediate_size,
                leaf_batch_size * leaf_seq_size,
                self.output_size,
            )),
            // Gradients are of size [B*S, I]
            // Gradient of intermediate activations (dl/dintermediate w1^T)
            ComputeUnit::single(
                Kernel::matmul(
                    "input act gradients: upstream * w1^T",
                    leaf_batch_size * leaf_seq_size,
                    leaf_intermediate_size,
                    self.input_size,
                ),
                strategy.collective(
                    "reduce scatter w2 gradients",
                    ShardingType::Data,
                    CollectiveType::ReduceScatter,
                    w2_bytes,
                ),
            ),
            // Gradient of w1 (input^T dl/dintermediate)
            ComputeUnit::single(
                Kernel::matmul(
                    "w1 gradient: input^T * upstream",
                    self.input_size,
                    leaf_batch_size * leaf_seq_size,
                    leaf_intermediate_size,
                ),
                strategy.collective(
                    "all-reduce input gradients",
                    ShardingType::Tensor,
                    CollectiveType::AllReduce,
                    input_act_bytes,
                ),
            ), // TODO: layernorm
        ];
        // reduce-scatter the w1 gradients in earlier layers
        let downstream_collective = strategy.collective(
            "reduce scatter w1 gradients",
            ShardingType::Data,
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
            weight_size: weight_memory,
            activation_size: activation_memory,
            cache_for_backprop: w1_act_memory + w2_act_memory,
            input_act_size: self.input_size * leaf_batch_size * leaf_seq_size,
            output_act_size: self.output_size * leaf_batch_size * leaf_seq_size,
        }
    }

    fn validate(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
    ) -> Result<(), ValidationError> {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();
        let batch_per_leaf = axes.batch / batch;
        let sequence_per_leaf = axes.sequence / sequence;
        let output_per_leaf = self.output_size / feature;

        if batch_per_leaf == 0 {
            return Err(ValidationError::InvalidBatchSplit(axes.batch, batch));
        } else if sequence_per_leaf == 0 {
            return Err(ValidationError::InvalidSequenceSplit(
                axes.sequence,
                sequence,
            ));
        } else if output_per_leaf == 0 {
            return Err(ValidationError::InvalidFeatureSplit(
                self.output_size,
                feature,
            ));
        } else if self.intermediate_size % feature != 0 {
            return Err(ValidationError::InvalidMLPSplit(
                self.intermediate_size,
                feature,
            ));
        }
        Ok(())
    }
}
