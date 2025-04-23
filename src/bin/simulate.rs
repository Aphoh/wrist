use wrist::{kernels, manual_models::naive_mlp::NaiveMLP, network::RegressionNetwork, sharding::SeqModelSpec, solver, tracing::Traceable};

fn main() {
    let axes = SeqModelSpec {
        batch: 16,
        sequence: 8192,
        feature: 4096,
        layers: 32,
    };
    let leaf_memory = 80e9 as u64; // 80 GB
    let n_tiers = 4;

    let model = NaiveMLP::new(&axes, leaf_memory);
    let kernel_profile = kernels::DenseLookupKernelProfile::from_file("csvs/kernel_profile.csv");
    //let kernel_profile = NaiveKernelProfile();
    let net = RegressionNetwork::from_file(n_tiers, "csvs/regression_strided_3.csv");
    let strategy = solver::DenseSolver::solve(&model, &net, &kernel_profile);
    if let Some(s) = strategy {
        println!("Strategy: {}", s);
        let trace = model.trace(&axes, &s, &net, &kernel_profile);
    } else {
        println!("No strategy found");
    }
}
