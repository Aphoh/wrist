use kernels::NaiveKernelProfile;
use models::transformer::DecoderTransformer;
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

fn main() {
    let axes = SeqModelSpec {
        batch: 16,
        sequence: 8192,
        feature: 4096,
        layers: 32,
    };
    let leaf_memory = 80e9 as u64; // 80 GB
    let n_tiers = 4;

    let tformer = DecoderTransformer::new(axes.clone(), 32, 8, leaf_memory);
    let kernel_profile = kernels::DenseLookupKernelProfile::from_file("kernel_profile.csv");
    //let kernel_profile = NaiveKernelProfile();
    let net = RegressionNetwork::from_file(n_tiers, "regression_strided_3.csv");
    let strategy = solver::DenseSolver::solve(&tformer, &net, &kernel_profile);
    if let Some(s) = strategy {
        println!("Strategy: {}", s);
        let trace = tformer.trace(&axes, &s, &net, &kernel_profile);
        let json_result = trace.to_json().unwrap();
        std::fs::write("trace.json", json_result).expect("Unable to write file");
    } else {
        println!("No strategy found");
    }
}
