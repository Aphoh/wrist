use crate::{
    network::Collective,
    sharding::{SeqModelSpec, ShardSpec, ShardStrategy, ShardingType},
};

pub trait Operation<M: ShardSpec> {
    fn compute_ms(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> u32;
    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> (u64, u64);
    fn micro_batch_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy<M>,
    ) -> Vec<Collective>;

    fn validate(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> bool;
    // TODO: deal with batch network ops
    //fn batch_network_ops(&self, axes: SeqModelSpec, strategy: &ShardStrategy<M>) -> Vec<Collective>;
}

pub struct Linear1DTpAllGather {
    pub input_size: u32,
    pub output_size: u32,
}

pub const FLOP_NS: f64 = 1e6;

impl<M: ShardSpec> Operation<M> for Linear1DTpAllGather {
    fn compute_ms(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> u32 {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let input_size = self.input_size;
        let batch_per_leaf = axes.batch / batch;
        let sequence_per_leaf = axes.sequence / sequence;
        let output_per_leaf = self.output_size / feature;

        debug_assert_eq!(input_size, axes.feature);
        // The operation is [B*S, F] x [F, O]
        // Because we have all the inputs, but only compute a subset of the output activations
        // so we have [B/b * S/s, F] x [F, O/f] which is 2 * B/b * S/s * F * O/f flops
        let flops = 2 * batch_per_leaf * sequence_per_leaf * input_size * output_per_leaf;
        return ((flops as f64 / FLOP_NS).ceil() as u32).max(1);
    }

    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> (u64, u64) {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let input_size = self.input_size;
        let batch_per_leaf = axes.batch / batch;
        let sequence_per_leaf = axes.sequence / sequence;
        let output_per_leaf = self.output_size / feature;

        let weight_memory = input_size as u64 * output_per_leaf as u64;
        let activation_memory = batch_per_leaf as u64 * sequence_per_leaf as u64 * self.output_size as u64;
        (weight_memory, activation_memory)
    }

    fn micro_batch_network_ops(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy<M>,
    ) -> Vec<Collective> {
        // We have to all-gather the output activations on each tier that uses tp
        // This should be done along the maximum tier axis with the correct number of pieces
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();
        let batch_per_leaf = axes.batch / batch;
        let sequence_per_leaf = axes.sequence / sequence;
        let output_per_leaf = self.output_size / feature;
        let leaf_act_size = batch_per_leaf * sequence_per_leaf * output_per_leaf;

        // Find the last tier that uses tp and calculate the number of leaf nodes
        if let Some(max_tier) = strategy
            .pieces
            .as_ref()
            .iter()
            .rposition(|&x| x == ShardingType::Tensor)
        {
            vec![Collective::all_gather(leaf_act_size, max_tier as u32)]
        } else {
            Default::default()
        }
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
        batch_per_leaf == 0 || sequence_per_leaf == 0 || output_per_leaf == 0
    }
    
}
