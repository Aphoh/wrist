use crate::{
    network::{CollectiveMeasurer, Network},
    ops::{Linear1DTpAllGather, Operation},
    sharding::{SeqModelSpec, ShardSpec, ShardStrategy, ShardingType},
    solver::Solveable,
};

pub struct NaiveMLP {
    axes: SeqModelSpec,
    leaf_memory: u64,
    ops: Vec<Linear1DTpAllGather>,
}

impl NaiveMLP {
    pub fn new(batch: u32, sequence: u32, feature: u32, layers: u32, leaf_memory: u64) -> Self {
        let axes = SeqModelSpec {
            batch,
            sequence,
            feature,
            layers,
        };
        let mut ops = Vec::new();
        for _ in 0..layers {
            ops.push(Linear1DTpAllGather {
                input_size: feature,
                output_size: feature,
            });
        }
        NaiveMLP {
            axes,
            ops,
            leaf_memory,
        }
    }

    // TODO: I should probably account for buffers for collectives too right?
    pub fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64> {
        let mut weight_mem = 0u64;
        let mut max_act_mem = 0u64;
        for op in &self.ops {
            if !op.validate(&self.axes, strategy) {
                return None;
            }
            let (w, a) = op.memory_bytes(&self.axes, strategy);
            weight_mem += w;
            max_act_mem = max_act_mem.max(a);
        }
        let max_mem = weight_mem + max_act_mem;
        (self.leaf_memory >= max_mem).then(|| max_mem)
    }

    pub fn forward_ms<S: ShardSpec, M: CollectiveMeasurer>(
        &self,
        strategy: &ShardStrategy<S>,
        network: &Network<M>,
    ) -> u32 {
        self.ops
            .iter()
            .map(|op| {
                let compute = op.compute_ms(&self.axes, strategy);
                let collectives = op.micro_batch_network_ops(&self.axes, strategy);
                let network_ms = network.duration_ms(collectives);
                return compute + network_ms;
            })
            .sum()
    }
}

impl Solveable for NaiveMLP {
    fn objective<S: ShardSpec, M: CollectiveMeasurer>(
        &self,
        strategy: &ShardStrategy<S>,
        network: &Network<M>,
    ) -> u32 {
        self.forward_ms(strategy, network)
    }

    fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64> {
        NaiveMLP::validate(self, strategy)
    }

    fn supported_shardings(&self) -> Vec<ShardingType> {
        vec![ShardingType::Data, ShardingType::Tensor]
    }
}
