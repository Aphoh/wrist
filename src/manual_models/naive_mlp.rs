use crate::{
    graph::{ComputeGraph, Subgraph},
    kernels::{Kernel, KernelProfile},
    network::{Collective, Network},
    sharding::{SeqModelSpec, ShardStrategy, ShardingType},
    solver::Solveable,
    tracing::Traceable,
    utils::ValidationError,
};

pub struct NaiveMLP {
    axes: SeqModelSpec,
    leaf_memory: u64,
    intermediate_size: u64,
}

impl NaiveMLP {
    pub fn new(axes: &SeqModelSpec, leaf_memory: u64) -> Self {
        NaiveMLP {
            axes: axes.clone(),
            leaf_memory,
            intermediate_size: axes.feature * 4,
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
    ) -> ComputeGraph {
        let (mut fwd, start) = Subgraph::new(axes.layers);
        let splits = strategy.axis_splits();
        let intermediate = self.intermediate_size / splits.feature;
        let batch = axes.batch * axes.sequence / splits.batch / splits.sequence;
        let fwd1 = fwd.kernel(
            [start],
            "intermediate",
            Kernel::matmul("w1", batch, axes.feature, intermediate),
            prof,
        );
        let fwd2 = fwd.kernel(
            [fwd1],
            "output",
            Kernel::matmul("w2", batch, intermediate, axes.feature),
            prof,
        );
        let bytes = 2 * batch * axes.feature;
        if let Some(coll) = strategy.collective(
            "activation reduce",
            ShardingType::Tensor,
            crate::network::CollectiveType::AllReduce,
            bytes,
        ) {
            let reduce = fwd.collective([fwd2], "reduce", coll, network);
            fwd.finish([reduce])
        } else {
            fwd.finish([fwd2]);
        }

        ComputeGraph {
            subgraphs: vec![fwd],
        }
    }
}

impl Solveable for NaiveMLP {
    fn objective(
        &self,
        strategy: &ShardStrategy,
        network: &impl Network,
        prof: &impl KernelProfile,
    ) -> u64 {
        self.trace(&self.axes, strategy, network, prof).time()
    }

    fn validate(&self, strategy: &ShardStrategy) -> Result<u64, ValidationError> {
        //TODO: Implement validation logic
        Ok(0)
    }

    fn supported_shardings(&self) -> Vec<ShardingType> {
        vec![ShardingType::Data, ShardingType::Tensor]
    }
}
