use crate::network::{Collective, CollectiveType};
use crate::ops::{MemoryProfile, Operation};
use crate::sharding::SeqModelSpec;
use crate::sharding::ShardSpec;
use crate::sharding::ShardStrategy;
use crate::sharding::ShardingType; // Add this line to import the Operation trait

pub struct DoubleLinearReductionParallel {
    pub input_size: u64,
    pub hidden_size: u64,
    pub output_size: u64,
}

pub const FLOP_US: f64 = 0.5 * 312e6;

impl<M: ShardSpec> Operation<M> for DoubleLinearReductionParallel {
    fn forward_us(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> u64 {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let leaf_hidden_size = self.hidden_size / feature;
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;

        let linear_1_flops =
            2 * leaf_batch_size * leaf_seq_size * self.input_size * leaf_hidden_size;
        let linear_2_flops =
            2 * leaf_batch_size * leaf_seq_size * leaf_hidden_size * self.output_size;
        let flops = linear_1_flops + linear_2_flops;
        return ((flops as f64 / FLOP_US).ceil() as u64).max(1);
    }

    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> MemoryProfile {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let leaf_hidden_size = self.hidden_size / feature;
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

    fn micro_batch_fwd_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy<M>,
    ) -> Vec<Collective> {
        // We have to all-reduce the output activations on each tier that uses tp
        // This should be done along the maximum tier axis with the correct number of pieces
        let SeqModelSpec {
            batch, sequence, ..
        } = strategy.axis_splits();
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;

        let output_act_size = 2 * leaf_batch_size * leaf_seq_size * self.output_size;
        strategy
            .collective(
                ShardingType::Tensor,
                CollectiveType::AllReduce,
                output_act_size,
            )
            .map(|c| vec![c])
            .unwrap_or_default()
    }

    fn validate(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> bool {
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

    fn backward_us(&self, _axes: &SeqModelSpec, _strategy: &ShardStrategy<M>) -> Option<u64> {
        None
    }

    fn micro_batch_bwd_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy<M>,
    ) -> Vec<Collective> {
        // We have to all-reduce the bwd gradients on each tier that uses tp
        // This should be done along the maximum tier axis with the correct number of pieces
        let SeqModelSpec {
            batch, sequence, ..
        } = strategy.axis_splits();
        let leaf_batch_size = axes.batch / batch;
        let leaf_seq_size = axes.sequence / sequence;

        let input_act_size = 2 * leaf_batch_size * leaf_seq_size * self.input_size;
        strategy
            .collective(
                ShardingType::Tensor,
                CollectiveType::AllReduce,
                input_act_size,
            )
            .map(|c| vec![c])
            .unwrap_or_default()
    }
}
