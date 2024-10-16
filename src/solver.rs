use crate::{
    combinations::{self, combinations_with_replacement},
    network::Network,
    sharding::{ShardSpec, ShardStrategy, ShardingType},
};
//use rayon::prelude::*;

pub trait Solveable {
    fn objective<S: ShardSpec, M: Network>(
        &self,
        strategy: &ShardStrategy<S>,
        network: &M,
    ) -> u64;
    fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64>;
    fn supported_shardings(&self) -> Vec<ShardingType>;
}

pub struct DenseSolver;
impl DenseSolver {
    pub fn solve<M: Network + Send + Sync, T: Solveable + Send + Sync>(
        model: &T,
        network: &M,
    ) -> Option<ShardStrategy<Vec<ShardingType>>> {
        let n_tiers = network.n_tiers();

        let supported_shardings = model.supported_shardings();
        // For now we only support data and tensor sharding
        debug_assert!(supported_shardings.contains(&ShardingType::Data));
        debug_assert!(supported_shardings.contains(&ShardingType::Tensor));
        debug_assert_eq!(supported_shardings.len(), 2);

        let n_combinations =
            combinations::n_combinations(n_tiers as usize, supported_shardings.len());
        println!("Evaluating n_combinations: {}", n_combinations);

        if let Some((strategy, max_mem, ns)) = combinations_with_replacement(
            &[ShardingType::Tensor, ShardingType::Data],
            n_tiers as usize,
        )
        .filter_map(|strategy| {
            let strategy = ShardStrategy::new(strategy).expect("Invalid strategy");
            model.validate(&strategy).map(|max_mem| {
                let obj = model.objective(&strategy, network);
                println!("Strategy: {}, objective: {}", strategy, obj);
                (strategy, max_mem, obj)
            })
        })
        .min_by(|a, b| a.2.cmp(&b.2))
        {
            println!("Best strategy: {}, max memory: {:.2}GB, ms: {}", strategy, (max_mem as f64) / 1e9, ns / 1000);
            Some(strategy)
        } else {
            None
        }
    }
}
