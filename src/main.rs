use network::RegressionNetwork;

pub mod combinations;
pub mod data;
pub mod models;
pub(crate) mod network;
pub mod ops;
pub mod sharding;
pub mod solver;
pub mod kernels;
pub mod utils;
mod tests;

fn main() {
    let micro_batch = 16;
    let sequence = 2048;
    let feature = 8192;
    let layers = 40;
    let leaf_memory = 80e9 as u64; // 80 GB
    let n_tiers = 4;

    let model_size = 2 * (feature as u64) * (feature as u64) * (layers as u64);
    println!("Model size: {:.2}GB", (model_size as f64) / 1e9);
    println!("Leaf memory: {:.2}GB", (leaf_memory as f64) / 1e9);

    let naive_mlp = models::naive_mlp::NaiveMLPForward::new(
        micro_batch,
        sequence,
        feature,
        layers,
        leaf_memory,
    );
    let net = RegressionNetwork::from_file(n_tiers, "regression_strided.csv");
    let strategy = solver::DenseSolver::solve(&naive_mlp, &net);
    if let Some(s) = strategy {
        println!("Strategy: {}", s);
    } else {
        println!("No strategy found");
    }
}
