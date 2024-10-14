use crate::{
    combinations::{self, combinations_with_replacement},
    network::{CollectiveMeasurer, Network},
    sharding::{ShardSpec, ShardStrategy, ShardingType},
};
use rayon::prelude::*;

pub trait Solveable {
    fn objective<S: ShardSpec, M: CollectiveMeasurer>(
        &self,
        strategy: &ShardStrategy<S>,
        network: &Network<M>,
    ) -> u32;
    fn validate<S: ShardSpec>(&self, strategy: &ShardStrategy<S>) -> Option<u64>;
    fn supported_shardings(&self) -> Vec<ShardingType>;
}

pub struct DenseSolver;
impl DenseSolver {
    pub fn solve<M: CollectiveMeasurer + Send + Sync, T: Solveable + Send + Sync>(
        model: &T,
        network: &Network<M>,
    ) -> Option<ShardStrategy<Vec<ShardingType>>> {
        let n_tiers = network.n_tiers();

        let supported_shardings = model.supported_shardings();
        // For now we only support data and tensor sharding
        debug_assert!(supported_shardings.contains(&ShardingType::Data));
        debug_assert!(supported_shardings.contains(&ShardingType::Tensor));
        debug_assert_eq!(supported_shardings.len(), 2);

        dbg!(n_tiers, supported_shardings.len());
        let n_combinations =
            combinations::n_combinations(n_tiers as usize, supported_shardings.len());
        println!("Evaluating n_combinations: {}", n_combinations);

        if let Some((strategy, max_mem, ms)) = combinations_with_replacement(
            &[ShardingType::Data, ShardingType::Tensor],
            n_tiers as usize,
        )
        .par_bridge()
        .filter_map(|strategy| {
            let strategy = ShardStrategy::new_unchecked(strategy);
            model.validate(&strategy).map(|max_mem| {
                let obj = model.objective(&strategy, network);
                (strategy, max_mem, obj)
            })
        })
        .min_by(|(_, _, a), (_, _, b)| a.cmp(b))
        {
            println!(
                "Found strategy with ms: {}, max_mem: {:.2}GB",
                ms,
                (max_mem as f64) / 1e9
            );
            Some(strategy)
        } else {
            None
        }
    }
}
