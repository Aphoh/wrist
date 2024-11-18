use crate::{
    combinations::{self, permuted_combinations_with_replacement},
    kernels::KernelProfile,
    network::Network,
    sharding::{ShardStrategy, ShardingType},
    utils::ValidationError,
};
//use rayon::prelude::*;

pub trait Solveable {
    fn objective(
        &self,
        strategy: &ShardStrategy,
        network: &impl Network,
        prof: &impl KernelProfile,
    ) -> u64;
    fn validate(&self, strategy: &ShardStrategy) -> Result<u64, ValidationError>;
    fn supported_shardings(&self) -> Vec<ShardingType>;
}

pub struct DenseSolver;
impl DenseSolver {
    pub fn solve<M: Network + Send + Sync, T: Solveable + Send + Sync>(
        model: &T,
        network: &M,
        kernel_profiler: &impl KernelProfile,
    ) -> Option<ShardStrategy> {
        let n_tiers = network.n_tiers();

        let supported_shardings = model.supported_shardings();

        let n_combinations =
            combinations::n_repeated_combinations(n_tiers as usize, supported_shardings.len());
        println!("Evaluating n_combinations: {}", n_combinations);

        if let Some((strategy, max_mem, ns)) =
            permuted_combinations_with_replacement(&supported_shardings, n_tiers as usize)
                .filter_map(|strategy| {
                    let strategy = ShardStrategy::new(strategy).expect("Invalid strategy");
                    model
                        .validate(&strategy)
                        .map(|max_mem| {
                            let obj = model.objective(&strategy, network, kernel_profiler);
                            println!("Strategy: {}, objective: {}", strategy, obj as f32 / 1e3);
                            (strategy, max_mem, obj)
                        })
                        .map_err(|e| {
                            println!("Invalid strategy: {}", e);
                            e
                        })
                        .ok()
                })
                .min_by(|a, b| a.2.cmp(&b.2))
        {
            println!(
                "Best strategy: {}, max memory: {:.2}GB, ms: {}",
                strategy,
                (max_mem as f64) / 1e9,
                ns / 1000
            );
            Some(strategy)
        } else {
            None
        }
    }
}
