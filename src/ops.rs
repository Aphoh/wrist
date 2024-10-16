use crate::{
    network::Collective,
    sharding::{SeqModelSpec, ShardSpec, ShardStrategy, ShardingType},
};

pub struct Profile {
    pub compute_ns: u64,
    pub static_memory: u64,
    pub peak_activation_memory: u64,
    pub micro_collectives: Vec<Collective>,
    pub batch_collectives: Vec<Collective>,
}

pub trait Operation<M: ShardSpec> {
    fn compute_us(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> u64;
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

pub const FLOP_US: f64 = 0.5 * 312e6;

impl<M: ShardSpec> Operation<M> for Linear1DTpAllGather {
    fn compute_us(&self, axes: &SeqModelSpec, strategy: &ShardStrategy<M>) -> u64 {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let input_size = self.input_size as u64;
        let batch_per_leaf = (axes.batch / batch) as u64;
        let sequence_per_leaf = (axes.sequence / sequence) as u64;
        let output_per_leaf = (self.output_size / feature) as u64;

        let m = batch_per_leaf * sequence_per_leaf;
        let k = input_size;
        let n = output_per_leaf;
        debug_assert_eq!(input_size, axes.feature as u64);
        // The operation is [B*S, F] x [F, O]
        // Because we have all the inputs, but only compute a subset of the output activations
        // so we have [B/b * S/s, F] x [F, O/f] which is 2 * B/b * S/s * F * O/f flops
        let flops = 2 * m * n * k;
        return ((flops as f64 / FLOP_US).ceil() as u64).max(1);
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

        let weight_memory = 2 * input_size as u64 * output_per_leaf as u64;
        let activation_memory =
            2 * batch_per_leaf as u64 * sequence_per_leaf as u64 * self.output_size as u64;
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
        let batch_per_leaf = (axes.batch / batch) as u64;
        let sequence_per_leaf = (axes.sequence / sequence) as u64;
        let output_per_leaf = (self.output_size / feature) as u64;
        let leaf_act_size = batch_per_leaf * sequence_per_leaf * output_per_leaf;

        // Find the last tier that uses tp and calculate the number of leaf nodes
        if let Some(tp_tier) = strategy.top_tier_for(ShardingType::Tensor)
        {
            // The network tier is indexed from 0 at the leaf tier
            vec![Collective::all_gather(leaf_act_size, tp_tier as u32)]
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
        batch_per_leaf != 0 && sequence_per_leaf != 0 && output_per_leaf != 0
    }
}
