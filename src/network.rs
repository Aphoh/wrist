use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Eq, PartialEq, Debug, Hash, Clone, Copy, PartialOrd, Ord)]
pub enum CollectiveType {
    AllGather,
    Gather,
    AllReduce,
    Reduce,
}

#[derive(Eq, PartialEq, Debug, Hash, Clone, PartialOrd, Ord)]
pub struct Collective {
    ctype: CollectiveType,
    piece_bytes: u64,
    tier: u32,
}

impl Collective {
    pub fn all_gather(piece_bytes: u64, tier: u32) -> Self {
        Self {
            ctype: CollectiveType::AllGather,
            piece_bytes,
            tier,
        }
    }

    pub fn reduce(piece_bytes: u64, tier: u32) -> Self {
        Self {
            ctype: CollectiveType::Reduce,
            piece_bytes,
            tier,
        }
    }
}
pub trait Network {
    fn measure<C: AsRef<[Collective]>>(&self, collectives: C) -> u64;
    fn n_tiers(&self) -> u32;

    fn accelerators_by_tier(&self) -> Vec<u32> {
        (0..self.n_tiers())
            .map(|a| self.num_accelerators(a))
            .collect()
    }

    fn num_accelerators(&self, tier: u32) -> u32 {
        return 2u32.pow(tier + 1);
    }
}

pub struct NaiveManualNetwork {
    n_tiers: u32,
}
impl NaiveManualNetwork {
    pub fn new(n_tiers: u32) -> Self {
        NaiveManualNetwork { n_tiers }
    }
}
impl Network for NaiveManualNetwork {
    fn measure<C: AsRef<[Collective]>>(&self, collectives: C) -> u64 {
        collectives
            .as_ref()
            .iter()
            .map(|c| match c.ctype {
                CollectiveType::AllGather => {
                    let latency = 100 * (self.n_tiers - c.tier);
                    let each_comm = (c.piece_bytes as f64) / ((1 << 27) as f64);
                    let all_comms = each_comm * ((self.n_tiers - c.tier + 1) as f64);
                    let rounded = all_comms.ceil() as u64;
                    latency as u64 + u64::try_from(rounded).expect("uh oh")
                }
                CollectiveType::Gather => 64 * 2u64.pow(c.tier),
                CollectiveType::AllReduce => 64 * 2u64.pow(c.tier),
                CollectiveType::Reduce => 64 * c.tier as u64,
            })
            .max() 
            .unwrap_or_default()
    }

    fn n_tiers(&self) -> u32 {
        self.n_tiers
    }
}

pub struct LogLogRegression {
    log_intercept: f64,
    log_coeff: f64,
    min: f64,
}

#[derive(Debug, Deserialize)]
pub struct RegressionEntry {
    gpu_group: u32,
    operation: String,
    log_coef: f64,
    log_intercept: f64,
    log_r_squared: f64,
    smallest_data_mean_time: f64,
    smallest_data_size: f64,
}

pub struct RegressionNetwork {
    regressions: BTreeMap<(CollectiveType, u32), LogLogRegression>,
    n_tiers: u32,
}

impl RegressionNetwork {
    pub fn from_file<P: AsRef<Path>>(n_tiers: u32, path: P) -> Self {
        let mut reader = csv::Reader::from_path(path).expect("Failed to read regression file");
        let mut regressions = BTreeMap::new();
        for record in reader.deserialize() {
            let record: RegressionEntry = record.expect("Failed to parse regression entry");
            let regression = LogLogRegression {
                log_intercept: record.log_intercept,
                log_coeff: record.log_coef,
                min: record.smallest_data_mean_time,
            };
            let ctype = match record.operation.as_str() {
                "all_gather" => CollectiveType::AllGather,
                "gather" => CollectiveType::Gather,
                "all_reduce" => CollectiveType::AllReduce,
                "reduce" => CollectiveType::Reduce,
                _ => panic!("Unknown operation type"),
            };
            let key = (ctype, record.gpu_group);
            regressions.insert(key, regression);
        }

        RegressionNetwork {
            n_tiers,
            regressions,
        }
    }
}

impl Network for RegressionNetwork {
    fn measure<C: AsRef<[Collective]>>(&self, collectives: C) -> u64 {
        let mut total_latency_us= 0u64;
        for c in collectives.as_ref() {
            let n_gpus = self.num_accelerators(c.tier);
            let regression = self.regressions.get(&(c.ctype, n_gpus)).unwrap();
            let piece_log = (c.piece_bytes as f64).log2();
            let latency_s = (regression.log_intercept + regression.log_coeff * piece_log)
                .exp2()
                .min(regression.min);
            debug_assert!(latency_s.is_finite() && latency_s > 0.0);
            //println!("{:?} ngpu: {} data: {: >4}mb {:.4}ms", c.ctype, n_gpus, c.piece_bytes / (1<<20), latency_s * 1e3);
            total_latency_us += (1e6 * latency_s) as u64;
        }
        total_latency_us
    }

    fn n_tiers(&self) -> u32 {
        self.n_tiers
    }
}
