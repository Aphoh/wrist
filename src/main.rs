use network::NaiveCollectiveMeasurer;

pub mod data;
mod tests;
pub mod models;
pub(crate) mod network;
pub mod ops;
pub mod sharding;
pub mod solver;
pub mod combinations;

fn main() {


    let batch = 2048;
    let sequence = 2048;
    let feature = 4096;
    let layers = 20;
    let leaf_memory = 1 << 34; // 0.5 GB
    let n_tiers = 9; // 16k nodes

    let naive_mlp = models::naive_mlp::NaiveMLP::new(batch, sequence, feature, layers, leaf_memory);
    let measurer = NaiveCollectiveMeasurer{};
    let network = network::Network::new(n_tiers, measurer);
    let strategy = solver::DenseSolver::solve(&naive_mlp, &network);
    println!("Strategy: {:?}", strategy);
}
