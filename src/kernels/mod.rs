use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KernelOp {
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
pub struct Kernel {
    pub name: String,
    pub op: KernelOp,
}

impl From<Kernel> for KernelOp {
    fn from(named: Kernel) -> Self {
        named.op
    }
}

impl AsRef<KernelOp> for Kernel {
    fn as_ref(&self) -> &KernelOp {
        &self.op
    }
}

impl AsRef<KernelOp> for KernelOp {
    fn as_ref(&self) -> &KernelOp {
        &self
    }
}

impl Kernel {
    pub fn new(name: impl ToString, kernel: KernelOp) -> Self {
        Kernel {
            name: name.to_string(),
            op: kernel,
        }
    }

    pub fn matmul(name: impl ToString, m: u64, k: u64, n: u64) -> Self {
        Kernel::new(name, KernelOp::MatMul { m, n, k })
    }

    pub fn flash_attention(
        name: impl ToString,
        b: u64,
        s: u64,
        kv_heads: u64,
        query_heads: u64,
        head_dim: u64,
    ) -> Self {
        Kernel::new(
            name,
            KernelOp::FlashAttentionFwd {
                b,
                s,
                kv_heads,
                query_heads,
                head_dim,
            },
        )
    }
}


pub trait KernelProfile {
    fn compute_us<I: AsRef<KernelOp>>(&self, kernel: I) -> u64;
}

pub struct NaiveKernelProfile();
pub const FLOPS_PER_US: f64 = 0.5 * 312e6;
// TODO: actually use data
impl KernelProfile for NaiveKernelProfile {
    fn compute_us<K: AsRef<KernelOp>>(&self, kernel: K) -> u64 {
        let flops = match *kernel.as_ref() {
            KernelOp::MatMul { m, n, k } => 2 * m * n * k,
            KernelOp::FlashAttentionFwd {
                b,
                s,
                query_heads,
                head_dim,
                ..
            } => flash_attention_forward_flops(b, s, query_heads, head_dim),
            KernelOp::LayerNorm { n, hidden } => 2 * n * hidden,
            KernelOp::FlashAttentionBwd {
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
    pub records: HashMap<KernelOp, u64>,
}

impl DenseLookupKernelProfile {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let mut reader = csv::Reader::from_path(path).expect("Failed to read kernel profile file");
        let mut records = HashMap::new();
        for row in reader.deserialize() {
            let row: ProfileRow = row.expect("Failed to parse row");
            let kernel = match row.ty.as_str() {
                "matmul" => KernelOp::MatMul {
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
    fn compute_us<K: AsRef<KernelOp>>(&self, kernel: K) -> u64 {
        let k = kernel.as_ref();
        self.records
            .get(&k)
            .map(|c| *c)
            .unwrap_or_else(|| NaiveKernelProfile().compute_us(k))
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
