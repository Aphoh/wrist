use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Eq, PartialEq, Debug, Hash, Clone, Copy, PartialOrd, Ord, Serialize)]
pub enum CollectiveType {
    AllGather,
    AllReduce,
    ReduceScatter,
    Ring,
}

const COLLECTIVES: [&'static str; 4] = ["all_gather", "all_reduce", "reduce_scatter", "ring"];

#[derive(Eq, PartialEq, Debug, Hash, Clone, PartialOrd, Ord, Serialize)]
pub struct Collective {
    pub ctype: CollectiveType,
    /// Stride is the spacing between leaves in the tree
    /// so for a 8 node tree, 3 tiers:
    /// Stride 2, Tier 2: [0, 2, 4, 6], [1, 3, 5, 7]
    /// Stride 1, Tier 2: [0, 1, 2, 3], [4, 5, 6, 7]
    /// Stride 4, Tier 1: [0, 4], [1, 5], [2, 6], [3, 7]
    /// Stride 2, Tier 1: [0, 2], [1, 3], [4, 6], [5, 7]
    /// Stride 1, Tier 1: [0, 1], [2, 3], [4, 5], [6, 7]
    pub stride: u32,
    pub piece_bytes: u64,
    pub n_gpus: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedCollective {
    pub name: String,
    pub collective: Collective,
}

impl AsRef<Collective> for NamedCollective {
    fn as_ref(&self) -> &Collective {
        &self.collective
    }
}

impl Collective {
    pub fn all_gather(piece_bytes: u64, tier: u32, stride: u32) -> Self {
        Self {
            ctype: CollectiveType::AllGather,
            stride,
            piece_bytes,
            n_gpus: tier,
        }
    }

    pub fn all_reduce(piece_bytes: u64, tier: u32, stride: u32) -> Self {
        Self {
            ctype: CollectiveType::AllReduce,
            stride,
            piece_bytes,
            n_gpus: tier,
        }
    }
}
pub trait Network {
    fn measure_maybe(&self, collective: &Option<impl AsRef<Collective>>) -> u64 {
        collective
            .as_ref()
            .map(|c| self.measure_one(c.as_ref()))
            .unwrap_or(0)
    }
    fn measure_one(&self, collective: &Collective) -> u64;
    fn n_tiers(&self) -> u32;
}

pub struct LogLogRegression {
    log_intercept: f64,
    log_coeff: f64,
    min: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct RegressionEntry {
    gpu_group: u32,
    operation: String,
    log_coef: f64,
    log_intercept: f64,
    log_r_squared: f64,
    smallest_data_mean_time: f64,
    smallest_data_size: f64,
    stride: u32,
}

impl RegressionEntry {
    fn key(&self) -> RegressionKey {
        RegressionKey {
            ctype: match self.operation.as_str() {
                "all_gather" => CollectiveType::AllGather,
                "all_reduce" => CollectiveType::AllReduce,
                "reduce_scatter" => CollectiveType::ReduceScatter,
                "ring" => CollectiveType::Ring,
                _ => panic!("Unknown operation type"),
            },
            stride: self.stride,
            n_gpus: self.gpu_group,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RegressionKey {
    ctype: CollectiveType,
    stride: u32,
    n_gpus: u32,
}

pub struct RegressionNetwork {
    regressions: BTreeMap<RegressionKey, LogLogRegression>,
    n_tiers: u32,
}

impl RegressionNetwork {
    pub fn from_file<P: AsRef<Path>>(n_tiers: u32, path: P) -> Self {
        let mut reader = csv::Reader::from_path(path).expect("Failed to read regression file");
        let mut regressions = BTreeMap::new();
        for record in reader.deserialize() {
            let record: RegressionEntry = record.expect("Failed to parse regression entry");
            if COLLECTIVES.contains(&record.operation.as_str()) {
                let regression = LogLogRegression {
                    log_intercept: record.log_intercept,
                    log_coeff: record.log_coef,
                    min: record.smallest_data_mean_time,
                };
                regressions.insert(record.key(), regression);
            }
        }

        RegressionNetwork {
            n_tiers,
            regressions,
        }
    }
}

impl Network for RegressionNetwork {
    fn n_tiers(&self) -> u32 {
        self.n_tiers
    }

    fn measure_one(&self, c: &Collective) -> u64 {
        let key = RegressionKey {
            ctype: c.ctype,
            stride: c.stride,
            n_gpus: c.n_gpus,
        };
        let regression = self.regressions.get(&key);
        if regression.is_none() {
            println!("No regression for {:?}", key);
            return 0;
        }
        let regression = regression.unwrap();
        // We go in units of megabytes = 2^20 bytes, so we should subtract 20 and max with 0.
        let piece_log = (c.piece_bytes as f64).log2() - 20.0;
        let piece_log = piece_log.max(0.0);
        let latency_s = (regression.log_intercept + regression.log_coeff * piece_log)
            .exp2()
            .max(regression.min);
        (1e6 * latency_s) as u64
    }
}
