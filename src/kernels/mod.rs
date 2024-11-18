use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kernel {
    MatMul {
        m: u64,
        k: u64,
        n: u64,
    },
    FlashAttentionFwd {
        b: u64,
        s: u64,
        kv_heads: u64,
        query_heads: u64,
        head_dim: u64,
    },
    FlashAttentionBwd {
        b: u64,
        s: u64,
        kv_heads: u64,
        query_heads: u64,
        head_dim: u64,
    },
    LayerNorm {
        n: u64,
        hidden: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedKernel {
    pub name: String,
    pub kernel: Kernel,
}

impl NamedKernel {
    pub fn new(name: impl ToString, kernel: Kernel) -> Self {
        NamedKernel {
            name: name.to_string(),
            kernel,
        }
    }

    pub fn matmul(name: impl ToString, m: u64, k: u64, n: u64) -> Self {
        NamedKernel::new(name, Kernel::MatMul { m, n, k })
    }

    pub fn flash_attention(
        name: impl ToString,
        b: u64,
        s: u64,
        kv_heads: u64,
        query_heads: u64,
        head_dim: u64,
    ) -> Self {
        NamedKernel::new(
            name,
            Kernel::FlashAttentionFwd {
                b,
                s,
                kv_heads,
                query_heads,
                head_dim,
            },
        )
    }
}

impl From<NamedKernel> for Kernel {
    fn from(named: NamedKernel) -> Self {
        named.kernel
    }
}

pub trait KernelProfile {
    fn compute_us<I: Into<Kernel>>(&self, kernel: I) -> u64;
}

pub struct NaiveKernelProfile();
pub const FLOPS_PER_US: f64 = 0.5 * 312e6;
// TODO: actually use data
impl KernelProfile for NaiveKernelProfile {
    fn compute_us<I: Into<Kernel>>(&self, kernel: I) -> u64 {
        let flops = match kernel.into() {
            Kernel::MatMul { m, n, k } => 2 * m * n * k,
            Kernel::FlashAttentionFwd {
                b,
                s,
                query_heads,
                head_dim,
                ..
            } => flash_attention_forward_flops(b, s, query_heads, head_dim),
            Kernel::LayerNorm { n, hidden } => 2 * n * hidden,
            Kernel::FlashAttentionBwd {
                b,
                s,
                query_heads,
                head_dim,
                ..
            } => 2 * flash_attention_forward_flops(b, s, query_heads, head_dim),
        };
        return (flops as f64 / FLOPS_PER_US) as u64;
    }
}

#[derive(Deserialize, Debug)]

struct ProfileRow {
    m: u64,
    n: u64,
    k: u64,
    time_ms: f64,
    #[serde(rename = "type")]
    ty: String,
}

pub struct DenseLookupKernelProfile {
    pub records: HashMap<Kernel, u64>,
}

impl DenseLookupKernelProfile {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let mut reader = csv::Reader::from_path(path).expect("Failed to read kernel profile file");
        let mut records = HashMap::new();
        for row in reader.deserialize() {
            let row: ProfileRow = row.expect("Failed to parse row");
            let kernel = match row.ty.as_str() {
                "matmul" => Kernel::MatMul {
                    m: row.m,
                    n: row.n,
                    k: row.k,
                },
                _ => panic!("Unknown kernel type"),
            };
            records.insert(kernel, (row.time_ms * 1e3) as u64);
        }

        Self { records }
    }
}

impl KernelProfile for DenseLookupKernelProfile {
    fn compute_us<I: Into<Kernel>>(&self, kernel: I) -> u64 {
        *self.records.get(&kernel.into()).expect("Kernel not found")
    }
}

fn flash_attention_forward_flops(
    batch_size: u64,
    sequence_length: u64,
    query_heads: u64,
    head_dim: u64,
) -> u64 {
    // Compute total FLOPs for forward pass
    let total_flops_qk =
        2 * batch_size * query_heads * sequence_length * sequence_length * head_dim;

    let total_flops_av =
        2 * batch_size * query_heads * sequence_length * sequence_length * head_dim;

    return total_flops_qk + total_flops_av;
}
