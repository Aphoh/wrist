use serde::Serialize;

use crate::{
    kernels::{KernelProfile, NamedKernel},
    network::{NamedCollective, Network},
    ops::ComputeUnit,
    sharding::{SeqModelSpec, ShardStrategy},
};

#[derive(Debug, Clone, Serialize)]
pub struct MeasuredKernel {
    kernel: NamedKernel,
    time_us: u64,
}

impl MeasuredKernel {
    pub fn measure(kernel: NamedKernel, prof: &impl KernelProfile) -> Self {
        let time_us = prof.compute_us(kernel.kernel);
        Self { kernel, time_us }
    }
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct MeasuredComputeUnit {
    kernels: Vec<MeasuredKernel>,
    collective: Option<MeasuredCollective>,
}

impl MeasuredComputeUnit {
    pub fn time_us(&self) -> u64 {
        let kernel_time_us: u64 = self.kernels.iter().map(|k| k.time_us).sum();
        let collective_time_us = self.collective.as_ref().map(|c| c.time_us).unwrap_or(0);
        kernel_time_us.max(collective_time_us)
    }

    pub fn from_collective(c: NamedCollective, network: &impl Network) -> Self {
        let measured = MeasuredCollective::measure(c, network);
        Self {
            kernels: Default::default(),
            collective: Some(measured),
        }
    }

    pub fn measure(cu: ComputeUnit, prof: &impl KernelProfile, network: &impl Network) -> Self {
        let kernels: Vec<_> = cu
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

#[derive(Debug, Clone, Serialize)]
pub enum TraceNode {
    Single {
        name: String,
        cus: Vec<MeasuredComputeUnit>,
    },
    Scan {
        name: String,
        cus: Vec<MeasuredComputeUnit>,
        n: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Trace {
    pub time_us: u64,
    pub nodes: Vec<TraceNode>,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
            time_us: 0,
        }
    }

    pub fn add(&mut self, node: TraceNode) {
        self.nodes.push(node);
    }

    pub fn measure_and_add(
        &mut self,
        name: impl ToString,
        cu: Vec<ComputeUnit>,
        prof: &impl KernelProfile,
        network: &impl Network,
    ) {
        let cus: Vec<_> = cu
            .into_iter()
            .map(|cu| MeasuredComputeUnit::measure(cu, prof, network))
            .collect();
        self.time_us += cus.iter().map(|cu| cu.time_us()).sum::<u64>();
        self.add(TraceNode::Single {
            name: name.to_string(),
            cus,
        });
    }

    pub fn measure_and_add_scan(
        &mut self,
        name: impl ToString,
        n: u64,
        cus: Vec<ComputeUnit>,
        prof: &impl KernelProfile,
        network: &impl Network,
    ) {
        let cus: Vec<_> = cus
            .into_iter()
            .map(|cu| MeasuredComputeUnit::measure(cu, prof, network))
            .collect();
        self.time_us += cus.iter().map(|cu| cu.time_us()).sum::<u64>() * n;
        self.add(TraceNode::Scan {
            name: name.to_string(),
            cus,
            n,
        });
    }

    pub fn measure_and_add_collective(
        &mut self,
        name: impl ToString,
        c: NamedCollective,
        network: &impl Network,
    ) {
        let measured = MeasuredComputeUnit::from_collective(c, network);
        self.time_us += measured.time_us();
        self.add(TraceNode::Single {
            name: name.to_string(),
            cus: vec![measured],
        });
    }

    pub fn time_us(&self) -> u64 {
        self.time_us
    }

    pub fn pretty_print(&self) -> String {
        let mut output = String::new();

        for (i, node) in self.nodes.iter().enumerate() {
            // Add separator between nodes
            if i > 0 {
                output.push_str("\n----------------------------------------\n");
            }

            match node {
                TraceNode::Single { cus, name } => {
                    output.push_str(&format!("{}: {} seq\n", i, name));
                    Self::format_compute_units(&mut output, cus);
                }
                TraceNode::Scan { name, cus, n } => {
                    output.push_str(&format!("{}: {} scan({})\n", i, name, n));
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

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self)
    }
}

pub trait Traceable {
    fn trace(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        network: &impl Network,
        kernel_profile: &impl KernelProfile,
    ) -> Trace;
}
