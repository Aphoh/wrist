use crate::{kernels::KernelProfile, network::Network, ops::ComputeUnit};

pub fn compute_us(
    compute_units: Vec<ComputeUnit>,
    network: &impl Network,
    kernel_profiler: &KernelProfile,
) -> u64 {
    let mut total = 0;
    for unit in compute_units {
        let kernel_time: u64 = unit
            .kernels
            .iter()
            .map(|k| kernel_profiler.compute_us(k.kernel))
            .sum();
        let network_time = network.measure_maybe(&unit.collective);
        total += kernel_time.max(network_time);
    }
    total
}
