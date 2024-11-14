#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    MatMul {
        m: u64,
        k: u64,
        n: u64,
    },
    FlashAttention {
        b: u64,
        h: u64,
        q: u64,
        s: u64,
        d: u64,
    },
    LayerNorm {
        n: u64,
        hidden: u64,
    },
}

#[derive(Debug, Clone)]
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

    pub fn flash_attention(name: impl ToString, b: u64, h: u64, q: u64, s: u64, d: u64) -> Self {
        NamedKernel::new(name, Kernel::FlashAttention { b, h, q, s, d })
    }
}

impl From<NamedKernel> for Kernel {
    fn from(named: NamedKernel) -> Self {
        named.kernel
    }
}

pub struct KernelProfile();

pub const FLOPS_PER_US: f64 = 0.5 * 312e6;

// TODO: actually use data
impl KernelProfile {
    pub fn compute_us<I: Into<Kernel>>(&self, kernel: I) -> u64 {
        let flops = match kernel.into() {
            Kernel::MatMul { m, n, k } => 2 * m * n * k,
            Kernel::FlashAttention { b, h, q, s, d } => 2 * b * h * q * s * d,
            Kernel::LayerNorm { n, hidden } => 2 * n * hidden,
        };
        return (flops as f64 / FLOPS_PER_US) as u64;
    }
}
