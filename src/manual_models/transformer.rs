use crate::{
    kernels::KernelProfile,
    network::Network,
    ops::{composite::DoubleOp, scan::ForwardBackwardStackModel, AttentionOp, MLP},
    sharding::{SeqModelSpec, ShardStrategy},
    solver::Solveable,
    tracing::{Trace, Traceable},
    utils::ValidationError,
};

pub struct DecoderTransformer {
    pub leaf_memory: u64,
    pub axes: SeqModelSpec,
    body: ForwardBackwardStackModel<DoubleOp<AttentionOp, MLP>>,
}

impl DecoderTransformer {
    pub fn new(axes: SeqModelSpec, n_q_heads: u64, n_kv_heads: u64, leaf_memory: u64) -> Self {
        let attention = AttentionOp {
            n_q_heads,
            n_kv_heads,
        };
        let mlp = MLP {
            input_size: axes.feature,
            intermediate_size: 4 * axes.feature,
            output_size: axes.feature,
        };
        let body = ForwardBackwardStackModel::new(DoubleOp {
            op1: attention,
            op2: mlp,
        });
        DecoderTransformer {
            leaf_memory,
            axes,
            body,
        }
    }
}

impl Traceable for DecoderTransformer {
    fn trace(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        kernel_profile: &impl KernelProfile,
    ) -> Trace {
        return self.body.trace(axes, strategy, network, kernel_profile);
    }
}

impl Solveable for DecoderTransformer {
    fn objective(
        &self,
        strategy: &ShardStrategy,
        network: &impl Network,
        prof: &impl KernelProfile,
    ) -> u64 {
        self.body
            .forward_backward_us(&self.axes, strategy, network, prof)
    }

    fn validate(&self, strategy: &ShardStrategy) -> Result<u64, ValidationError> {
        self.body.validate(&self.axes, strategy, self.leaf_memory)
    }

    fn supported_shardings(&self) -> Vec<crate::sharding::ShardingType> {
        use crate::sharding::ShardingType::*;
        vec![Data, Tensor] // TODO: add more
    }
}
