pub enum Kernel {
    MatMul {
        m: u64,
        n: u64,
        k: u64,
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

pub struct KernelProfile();

pub const FLOPS_PER_NS: f64 = 0.5 * 312e6;

// TODO: actually use data
impl KernelProfile {
    pub fn compute_ns(&self, kernel: &Kernel) -> u64 {
        let flops = match kernel {
            Kernel::MatMul { m, n, k } => 2 * m * n * k,
            Kernel::FlashAttention { b, h, q, s, d } => 2 * b * h * q * s * d, 
            Kernel::LayerNorm { n, hidden } => 2 * n * hidden,
        };
        return (flops as f64 / FLOPS_PER_NS) as u64;
    }
}