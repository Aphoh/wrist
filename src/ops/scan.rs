use crate::{
    kernels::KernelProfile,
    network::{CollectiveType, Network},
    sharding::{SeqModelSpec, ShardStrategy, ShardingType},
    utils,
};

use super::Operation;

pub struct ForwardBackwardStackModel<Op> {
    op: Op,
}

impl<Op: Operation> ForwardBackwardStackModel<Op> {
    pub fn new(op: Op) -> Self {
        Self { op }
    }

    pub fn validate(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        leaf_memory: u64,
    ) -> Option<u64> {
        let SeqModelSpec {
            layers: pp_split,
            batch: dp_split,
            ..
        } = strategy.axis_splits();
        if axes.layers % pp_split != 0 || axes.layers / pp_split == 0 {
            return None;
        }
        let n_layers = axes.layers / pp_split;

        let mut profile = self.op.memory_bytes(axes, strategy);
        profile.gradient_size *= n_layers;
        profile.cache_for_backprop *= n_layers;
        profile.weight_memory *= n_layers;
        let optimizer_size = 4 * profile.weight_memory / dp_split; // Zero-2 sharding
        if profile.total() + optimizer_size > leaf_memory {
            return None;
        }
        self.op.validate(axes, strategy).then(|| profile.total())
    }

    pub fn forward_us(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
    ) -> u64 {
        let n_layers = axes.layers;

        // Forward pass
        let (tail_compute, downstream) = self.op.forward(axes, strategy, None);
        let tail_time = utils::compute_us(tail_compute, network, &KernelProfile());

        if n_layers > 1 {
            let (body_compute, _) = self.op.forward(axes, strategy, downstream);
            let body_time = utils::compute_us(body_compute, network, &KernelProfile())
            tail_time + (n_layers - 1) * body_time
        } else {
            tail_time
        }
    }

    pub fn forward_backward_us(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
    ) -> u64 {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            layers,
        } = strategy.axis_splits();
        let n_layers = axes.layers;

        // Forward pass
        let (tail_compute, downstream) = self.op.forward(axes, strategy, None);

        let (body_compute, _) = self.op.forward(axes, strategy, downstream);

        let tail_time = utils::compute_us(tail_compute, network, &KernelProfile());
        let body_time = utils::compute_us(body_compute, network, &KernelProfile());

        // Backward pass

        // Assume we can overlap the head downstream collective with data loading or something

        let op_fwd = self.op.forward_us(axes, strategy);
        let op_bwd = self.op.backward_us(axes, strategy).unwrap_or(2 * op_fwd);
        let op_comm = network.measure(self.op.micro_batch_fwd_network_ops(axes, strategy));
        let op_bwd_comm = network.measure(self.op.micro_batch_bwd_network_ops(axes, strategy));

        let memory_profile = self.op.memory_bytes(axes, strategy);

        // Assume the pipeline bubble is 0, since we're only doing inference
        let comm = strategy
            .collective(
                ShardingType::Pipeline,
                CollectiveType::Ring,
                memory_profile.activation_memory,
            )
            .map(|c| network.measure(&[c]))
            .unwrap_or(0);

        return (n_layers * (op_fwd + op_comm + op_bwd + op_bwd_comm)) + comm;
    }
}

pub struct ForwardStackModel<Op> {
    op: Op,
}

impl<Op> ForwardStackModel<Op>
where
    Op: Operation,
{
    pub fn new(op: Op) -> Self {
        Self { op }
    }

    pub fn validate(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        leaf_memory: u64,
    ) -> Option<u64> {
        let pp_split = strategy.axis_splits().layers;
        if axes.layers % pp_split != 0 || axes.layers / pp_split == 0 {
            return None;
        }
        let n_layers = axes.layers / pp_split;

        let mut profile = self.op.memory_bytes(axes, strategy);
        profile.gradient_size *= 0; // No backprop
        profile.cache_for_backprop *= 0; // No backprop
        profile.weight_memory *= n_layers;
        if profile.total() > leaf_memory {
            return None;
        }
        self.op.validate(axes, strategy).then(|| profile.total())
    }

    pub fn forward_us<M: Network>(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &M,
    ) -> u64 {
        let pp_split = strategy.axis_splits().layers;
        let n_layers = axes.layers / pp_split;
        let op_fwd = self.op.forward_us(axes, strategy);
        let op_comm = network.measure(self.op.micro_batch_fwd_network_ops(axes, strategy));

        let memory_profile = self.op.memory_bytes(axes, strategy);

        // Assume the pipeline bubble is 0, since we're only doing inference
        let comm = strategy
            .collective(
                ShardingType::Pipeline,
                CollectiveType::Ring,
                memory_profile.activation_memory,
            )
            .map(|c| network.measure(&[c]))
            .unwrap_or(0);

        return (n_layers * (op_fwd + op_comm)) + comm;
    }

    pub fn forward_backward_us<M: Network>(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &M,
    ) {
        let fwd_ms = self.forward_us(axes, strategy, network);
    }
}
