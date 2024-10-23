use crate::{
    network::Network,
    ops::{DoubleLinearReductionParallel, MemoryProfile, Operation},
    sharding::{SeqModelSpec, ShardSpec, ShardStrategy, ShardingType},
    solver::Solveable,
};

pub struct NaiveMLPForward {
    axes: SeqModelSpec,
    leaf_memory: u64,
    ops: Vec<DoubleLinearReductionParallel>,
}

impl NaiveMLPForward {
    pub fn new(batch: u64, sequence: u64, feature: u64, layers: u64, leaf_memory: u64) -> Self {
        let axes = SeqModelSpec {
            batch,
            sequence,
            feature,
            layers,
        };
        let mut ops = Vec::new();
        for _ in 0..layers {
            ops.push(DoubleLinearReductionParallel {
                input_size: feature,
                hidden_size: 4 * feature,
                output_size: feature,
            });
        }
        NaiveMLPForward {
            axes,
            ops,
            leaf_memory,
        }
    }

    // TODO: I should probably account for buffers for collectives too right?
    pub fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64> {
        let mut profile = MemoryProfile::default();
        for op in &self.ops {
            if !op.validate(&self.axes, strategy) {
                println!("Invalid strategy {} for op", strategy);
                return None;
            }
            profile = op.memory_bytes(&self.axes, strategy).combine(&profile)
        }
        let max_mem = profile.total();
        if self.leaf_memory < max_mem {
            println!(
                "Invalid strategy {}, exceeds leaf memory: {:.2}GB",
                strategy,
                (max_mem as f64) / 1e9
            );
            return None;
        } else {
            println!(
                "Valid strategy {}, max memory: {:.2}GB",
                strategy,
                (max_mem as f64) / 1e9
            );
            return Some(max_mem);
        }
    }

    pub fn micro_batch_us<S: ShardSpec, M: Network>(
        &self,
        strategy: &ShardStrategy<S>,
        network: &M,
    ) -> u64 {
        let mut forwards = vec![];
        self.ops
            .iter()
            .map(|op| {
                let compute = op.forward_us(&self.axes, strategy);
                forwards.push(compute);
                let collectives = op.micro_batch_fwd_network_ops(&self.axes, strategy);
                let network_us = network.measure(collectives);
                compute + network_us
            })
            .sum::<u64>()
    }
}

impl Solveable for NaiveMLPForward {
    fn objective<S: ShardSpec, M: Network>(&self, strategy: &ShardStrategy<S>, network: &M) -> u64 {
        self.micro_batch_us(strategy, network)
    }

    fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64> {
        NaiveMLPForward::validate(self, strategy)
    }

    fn supported_shardings(&self) -> Vec<ShardingType> {
        vec![ShardingType::Data, ShardingType::Tensor]
    }
}
