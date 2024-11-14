use crate::{
    kernels::{KernelProfile, NamedKernel},
    network::{NamedCollective, Network},
    ops::ComputeUnit,
    sharding::{SeqModelSpec, ShardStrategy},
};

#[derive(Debug, Clone)]
pub struct MeasuredKernel {
    kernel: NamedKernel,
    time_us: u64,
}

impl MeasuredKernel {
    pub fn measure(kernel: NamedKernel, prof: &KernelProfile) -> Self {
        let time_us = prof.compute_us(kernel.kernel);
        Self { kernel, time_us }
    }
}

#[derive(Debug, Clone)]
pub struct MeasuredCollective {
    collective: NamedCollective,
    time_us: u64,
}

impl MeasuredCollective {
    pub fn measure(collective: NamedCollective, network: &impl Network) -> Self {
        let time_us = network.measure_one(&collective.collective);
        Self {
            collective,
            time_us,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeasuredComputeUnit {
    kernels: Vec<MeasuredKernel>,
    collective: Option<MeasuredCollective>,
}

impl MeasuredComputeUnit {
    pub fn from_collective(c: NamedCollective, network: &impl Network) -> Self {
        let collective = Some(MeasuredCollective::measure(c, network));
        Self {
            kernels: Default::default(),
            collective,
        }
    }

    pub fn measure(cu: ComputeUnit, prof: &KernelProfile, network: &impl Network) -> Self {
        let kernels = cu
            .kernels
            .into_iter()
            .map(|k| MeasuredKernel::measure(k, prof))
            .collect();
        let collective = cu
            .collective
            .map(|c| MeasuredCollective::measure(c, network));
        Self {
            kernels,
            collective,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TraceNode {
    Single(Vec<MeasuredComputeUnit>),
    Scan {
        cus: Vec<MeasuredComputeUnit>,
        n: u64,
    },
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub nodes: Vec<TraceNode>,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    pub fn add(&mut self, node: TraceNode) {
        self.nodes.push(node);
    }

    pub fn measure_and_add(
        &mut self,
        cu: Vec<ComputeUnit>,
        prof: &KernelProfile,
        network: &impl Network,
    ) {
        let measured = cu
            .into_iter()
            .map(|cu| MeasuredComputeUnit::measure(cu, prof, network))
            .collect();
        self.add(TraceNode::Single(measured));
    }

    pub fn measure_and_add_scan(
        &mut self,
        n: u64,
        cus: Vec<ComputeUnit>,
        prof: &KernelProfile,
        network: &impl Network,
    ) {
        let measured = cus
            .into_iter()
            .map(|cu| MeasuredComputeUnit::measure(cu, prof, network))
            .collect();
        self.add(TraceNode::Scan { cus: measured, n });
    }

    pub fn measure_and_add_collective(&mut self, c: NamedCollective, network: &impl Network) {
        let measured = MeasuredComputeUnit::from_collective(c, network);
        self.add(TraceNode::Single(vec![measured]));
    }

    // Need to fill this in!
    pub fn pretty_print(&self) -> String {
        let mut output = String::new();

        for (i, node) in self.nodes.iter().enumerate() {
            // Add separator between nodes
            if i > 0 {
                output.push_str("\n----------------------------------------\n");
            }

            match node {
                TraceNode::Single(cus) => {
                    output.push_str(&format!("Node {}: Sequential Execution\n", i));
                    Self::format_compute_units(&mut output, cus);
                }
                TraceNode::Scan { cus, n } => {
                    output.push_str(&format!("Node {}: Scan (repeated {} times)\n", i, n));
                    Self::format_compute_units(&mut output, cus);
                }
            }
        }

        output
    }

    fn format_compute_units(output: &mut String, cus: &[MeasuredComputeUnit]) {
        for (i, cu) in cus.iter().enumerate() {
            output.push_str(&format!("\nCompute Unit {}:\n", i));

            // Calculate total kernel time
            let total_kernel_time: u64 = cu.kernels.iter().map(|k| k.time_us).sum();

            // Format kernels
            if !cu.kernels.is_empty() {
                output.push_str("  Kernels (sequential):\n");
                for kernel in &cu.kernels {
                    output.push_str(&format!(
                        "    {} - {} μs\n",
                        kernel.kernel.name, kernel.time_us
                    ));
                }
                output.push_str(&format!("  Total kernel time: {} μs\n", total_kernel_time));
            }

            // Format collective if present
            if let Some(collective) = &cu.collective {
                output.push_str(&format!(
                    "  Collective (parallel with kernels):\n    {} - {} μs\n",
                    collective.collective.name, collective.time_us
                ));

                // Show actual execution time (max of kernel and collective times)
                let total_time = total_kernel_time.max(collective.time_us);
                output.push_str(&format!("  Effective execution time: {} μs\n", total_time));
            } else if !cu.kernels.is_empty() {
                output.push_str(&format!(
                    "  Effective execution time: {} μs\n",
                    total_kernel_time
                ));
            }
        }
    }
}

pub trait Traceable {
    fn trace(&self, axes: &SeqModelSpec, strategy: &ShardStrategy, network: &impl Network)
        -> Trace;
}
