use crate::{
    kernels::KernelProfile, network::Network, ops::{scan::ForwardBackwardStackModel, MLP}, sharding::{SeqModelSpec, ShardStrategy, ShardingType}, solver::Solveable, tracing::Traceable
};

pub struct NaiveMLP {
    axes: SeqModelSpec,
    leaf_memory: u64,
    model: ForwardBackwardStackModel<MLP>,
}

impl NaiveMLP {
    pub fn new(axes: &SeqModelSpec, leaf_memory: u64) -> Self {
        NaiveMLP {
            axes: axes.clone(),
            model: ForwardBackwardStackModel::new(MLP {
                input_size: axes.feature,
                intermediate_size: 4 * axes.feature,
                output_size: axes.feature,
            }),
            leaf_memory,
        }
    }
}

impl Traceable for NaiveMLP {
    fn trace(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        prof: &impl KernelProfile,
    ) -> crate::tracing::Trace {
        return self.model.trace(axes, strategy, network, prof);
    }
}

impl Solveable for NaiveMLP {
    fn objective(&self, strategy: &ShardStrategy, network: &impl Network, prof: &impl KernelProfile) -> u64 {
        self.model
            .forward_backward_us(&self.axes, strategy, network, prof)
    }

    fn validate(&self, strategy: &ShardStrategy) -> Option<u64> {
        self.model.validate(&self.axes, strategy, self.leaf_memory)
    }

    fn supported_shardings(&self) -> Vec<ShardingType> {
        vec![ShardingType::Data, ShardingType::Tensor]
    }
}
