use crate::{
    kernels::{KernelProfile, NaiveKernelProfile},
    network::Network,
    sharding::{SeqModelSpec, ShardStrategy},
    tracing::{Trace, Traceable},
    utils::{self, ValidationError},
};

use super::Operation;

pub struct ForwardBackwardStackModel<Op> {
    op: Op,
}

impl<Op: Operation> Traceable for ForwardBackwardStackModel<Op> {
    fn trace(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        kernel_profile: &impl KernelProfile,
    ) -> crate::tracing::Trace {
        // Forward Trace
        let mut trace = Trace::new();
        let (tail_units, tail_collective) = self.op.forward(axes, strategy, None);
        if let Some(collective) = &tail_collective {
            trace.measure_and_add_collective("pre fwd collective", collective.clone(), network);
        }
        let (body_units, _) = self.op.forward(axes, strategy, tail_collective);

        if axes.layers > 1 {
            trace.measure_and_add_scan("fwd main", axes.layers - 1, body_units, kernel_profile, network);
        }

        trace.measure_and_add("fwd tail", tail_units, kernel_profile, network);
        //Backward Trace
        let (tail_units, tail_collective) = self.op.backward(axes, strategy, None);
        let (body_units, body_collective) = self.op.backward(axes, strategy, tail_collective);
        trace.measure_and_add("bwd tail", tail_units, kernel_profile, network);

        if axes.layers > 1 {
            trace.measure_and_add_scan("bwd main", axes.layers - 1, body_units, kernel_profile, network);
        }

        if let Some(collective) = &body_collective {
            trace.measure_and_add_collective("post bwd collective", collective.clone(), network);
        }

        trace
    }
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
    ) -> Result<u64, ValidationError> {
        let SeqModelSpec {
            layers: pp_split,
            batch: dp_split,
            ..
        } = strategy.axis_splits();
        if axes.layers % pp_split != 0 || axes.layers / pp_split == 0 {
            return Err(ValidationError::InvalidLayerSplit(axes.layers, pp_split));
        }
        let n_layers = axes.layers / pp_split;

        let mut profile = self.op.memory_bytes(axes, strategy);
        profile.gradient_size *= n_layers;
        profile.cache_for_backprop *= n_layers;
        profile.weight_memory *= n_layers;
        let optimizer_size = 4 * profile.weight_memory / dp_split; // Zero-2 sharding
        let total = profile.total() + optimizer_size;
        if total > leaf_memory {
            return Err(ValidationError::InsufficientMemory(leaf_memory, total));
        }
        self.op.validate(axes, strategy).map(|_| total)
    }

    pub fn forward_backward_us(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        kernel_profile: &impl KernelProfile,
    ) -> u64 {
        self.trace(axes, strategy, network, kernel_profile).time_us
    }
}
