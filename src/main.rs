use network::RegressionNetwork;
use sharding::SeqModelSpec;
use tracing::Traceable;

pub mod combinations;
pub mod data;
pub mod kernels;
pub mod models;
pub(crate) mod network;
pub mod ops;
pub mod sharding;
pub mod solver;
mod tests;
pub mod tracing;
pub mod utils;
use serde_json;

fn main() {
    let axes = SeqModelSpec {
        batch: 16,
        sequence: 2048,
        feature: 8192,
        layers: 40,
    };
    let leaf_memory = 80e9 as u64; // 80 GB
    let n_tiers = 4;

    let model_size = 2 * (axes.feature as u64) * (axes.feature as u64) * (axes.layers as u64);
    println!("Model size: {:.2}GB", (model_size as f64) / 1e9);
    println!("Leaf memory: {:.2}GB", (leaf_memory as f64) / 1e9);

    let naive_mlp = models::naive_mlp::NaiveMLP::new(&axes, leaf_memory);
    let net = RegressionNetwork::from_file(n_tiers, "regression_strided_2.csv");
    let strategy = solver::DenseSolver::solve(&naive_mlp, &net);
    if let Some(s) = strategy {
        println!("Strategy: {}", s);
        let trace = naive_mlp.trace(&axes, &s, &net);
        println!("Trace: {}", trace.pretty_print());
        let json_result = trace.to_json().unwrap();
        std::fs::write("trace.json", json_result).expect("Unable to write file");
    } else {
        println!("No strategy found");
    }
}
