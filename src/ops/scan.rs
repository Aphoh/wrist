use crate::{
    kernels::KernelProfile,
    network::Network,
    sharding::{SeqModelSpec, ShardStrategy},
    tracing::{Trace, Traceable},
    utils,
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
    ) -> crate::tracing::Trace {
        let prof = KernelProfile();
        // Forward Trace
        let mut trace = Trace::new();
        let (tail_units, tail_collective) = self.op.forward(axes, strategy, None);
        if let Some(collective) = &tail_collective {
            trace.measure_and_add_collective("pre fwd collective", collective.clone(), network);
        }
        let (body_units, _) = self.op.forward(axes, strategy, tail_collective);

        if axes.layers > 1 {
            trace.measure_and_add_scan("fwd main", axes.layers - 1, body_units, &prof, network);
        }

        trace.measure_and_add("fwd tail", tail_units, &prof, network);
        //Backward Trace
        let (tail_units, tail_collective) = self.op.backward(axes, strategy, None);
        let (body_units, body_collective) = self.op.backward(axes, strategy, tail_collective);
        trace.measure_and_add("bwd tail", tail_units, &prof, network);

        if axes.layers > 1 {
            trace.measure_and_add_scan("bwd main", axes.layers - 1, body_units, &prof, network);
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

    pub fn forward_backward_us(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
    ) -> u64 {
        self.trace(axes, strategy, network).time_us
    }
}
