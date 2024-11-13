use crate::{
    network::Network,
    ops::{scan::ForwardStackModel, DoubleLinearReductionParallel},
    sharding::{SeqModelSpec, ShardStrategy, ShardingType},
    solver::Solveable,
};

pub struct NaiveMLPForward {
    axes: SeqModelSpec,
    leaf_memory: u64,
    model: ForwardStackModel<DoubleLinearReductionParallel>,
}

impl NaiveMLPForward {
    pub fn new(batch: u64, sequence: u64, feature: u64, layers: u64, leaf_memory: u64) -> Self {
        let axes = SeqModelSpec {
            batch,
            sequence,
            feature,
            layers,
        };
        NaiveMLPForward {
            axes,
            model: ForwardStackModel::new(DoubleLinearReductionParallel {
                input_size: feature,
                hidden_size: 4 * feature,
                output_size: feature,
            }),
            leaf_memory,
        }
    }
}

impl Solveable for NaiveMLPForward {
    fn objective<M: Network>(&self, strategy: &ShardStrategy, network: &M) -> u64 {
        self.model.forward_us(&self.axes, strategy, network)
    }

    fn validate(&self, strategy: &ShardStrategy) -> Option<u64> {
        self.model.validate(&self.axes, strategy, self.leaf_memory)
    }

    fn supported_shardings(&self) -> Vec<ShardingType> {
        vec![ShardingType::Data, ShardingType::Tensor, ShardingType::Pipeline]
    }
}
