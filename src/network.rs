use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Eq, PartialEq, Debug, Hash, Clone, Copy, PartialOrd, Ord, Serialize)]
pub enum CollectiveType {
    AllGather,
    AllReduce,
    ReduceScatter,
    Ring,
    AllToAllSingle,
}

const COLLECTIVES: [&'static str; 4] = ["all_gather", "all_reduce", "reduce_scatter", "ring"];

#[derive(Eq, PartialEq, Debug, Hash, Clone, PartialOrd, Ord, Serialize)]
pub struct Collective {
    pub name: String,
    pub ctype: CollectiveType,
    /// Stride is the spacing between leaves in the tree
    /// so for a 8 node tree, 3 tiers:
    /// Stride 2, Tier 2: [0, 2, 4, 6], [1, 3, 5, 7]
    /// Stride 1, Tier 2: [0, 1, 2, 3], [4, 5, 6, 7]
    /// Stride 4, Tier 1: [0, 4], [1, 5], [2, 6], [3, 7]
    /// Stride 2, Tier 1: [0, 2], [1, 3], [4, 6], [5, 7]
    /// Stride 1, Tier 1: [0, 1], [2, 3], [4, 5], [6, 7]
    pub group_stride: u32,
    pub piece_bytes: u64,
    pub group_size: u32,
}

impl Collective {
    pub fn all_gather(name: String, piece_bytes: u64, group_size: u32, stride: u32) -> Self {
        Self {
            name,
            ctype: CollectiveType::AllGather,
            group_stride: stride,
            piece_bytes,
            group_size,
        }
    }

    pub fn all_reduce(name: String, piece_bytes: u64, tier: u32, stride: u32) -> Self {
        Self {
            name,
            ctype: CollectiveType::AllReduce,
            group_stride: stride,
            piece_bytes,
            group_size: tier,
        }
    }

    pub fn reduce_scatter(
        name: String,
        piece_bytes: u64,
        group_size: u32,
        group_stride: u32,
    ) -> Collective {
        Self {
            name,
            ctype: CollectiveType::ReduceScatter,
            group_stride,
            piece_bytes,
            group_size,
        }
    }
}
pub trait Network: Sync {
    fn measure(&self, collective: &Collective) -> Option<u64>;
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
}

impl RegressionNetwork {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
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

        RegressionNetwork { regressions }
    }
}

impl Network for RegressionNetwork {
    fn measure(&self, c: &Collective) -> Option<u64> {
        let key = RegressionKey {
            ctype: c.ctype,
            stride: c.group_stride,
            n_gpus: c.group_size,
        };
        let regression = self.regressions.get(&key)?;
        // We go in units of megabytes = 2^20 bytes, so we should subtract 20 and max with 0.
        let piece_log = (c.piece_bytes as f64).log2() - 20.0;
        let piece_log = piece_log.max(0.0);
        let latency_s = (regression.log_intercept + regression.log_coeff * piece_log)
            .exp2()
            .max(regression.min);
        Some((1e6 * latency_s) as u64)
    }
}

/// A simple network model that assumes constant latency and bandwidth between nodes.
pub struct NaiveNetwork {
    /// Base latency in microseconds for any communication
    latency_us: f64,
    /// Bandwidth in bytes per microsecond
    bandwidth_bytes_per_us: f64,
}

impl NaiveNetwork {
    /// Create a new NaiveNetwork with the given latency (in microseconds) and bandwidth (in GB/s)
    pub fn new(latency_us: f64, bandwidth_gb_per_s: f64) -> Self {
        // Convert GB/s to bytes per microsecond (MB/s)
        let bandwidth_bytes_per_us = bandwidth_gb_per_s * 1000.0;
        Self {
            latency_us,
            bandwidth_bytes_per_us,
        }
    }
}

impl Network for NaiveNetwork {
    fn measure(&self, c: &Collective) -> Option<u64> {
        let n = c.group_size as f64;
        let data_bytes = c.piece_bytes as f64;
        let latency_us = self.latency_us as f64;

        // Base transfer time for the data
        let transfer_time = |bytes: f64| -> f64 { bytes / self.bandwidth_bytes_per_us };

        let time_us: f64 = match c.ctype {
            CollectiveType::AllGather => {
                // Each node receives data from all other nodes
                latency_us + transfer_time(data_bytes * (n - 1.0))
            }
            CollectiveType::AllReduce => {
                // Can be modeled as a reduce-scatter followed by all-gather
                // First, reduce-scatter phase (each node sends/receives (n-1)/n of data)
                let reduce_scatter_time = latency_us + transfer_time(data_bytes * (n - 1.0) / n);
                // Then, all-gather phase (each node sends/receives (n-1)/n of data)
                let all_gather_time = latency_us + transfer_time(data_bytes * (n - 1.0) / n);
                // Total time is the sum
                reduce_scatter_time + all_gather_time
            }
            CollectiveType::ReduceScatter => {
                // Each node sends (n-1)/n of its data and receives 1/n of data from every other node
                latency_us + transfer_time(data_bytes * (n - 1.0) / n)
            }
            CollectiveType::Ring => {
                // In a ring algorithm, each node sends/receives data in n-1 steps
                latency_us * (n - 1.0) + transfer_time(data_bytes * 2.0 * (n - 1.0) / n)
            }
            CollectiveType::AllToAllSingle => {
                // Each node sends a distinct piece of data to every other node
                latency_us + transfer_time(data_bytes * (n - 1.0))
            }
        };

        Some(time_us as u64)
    }
}

/// Enum wrapper for Network implementations to enable static dispatch
pub enum NetworkImpl {
    Regression(RegressionNetwork),
    Naive(NaiveNetwork),
}

impl NetworkImpl {
    /// Create a NetworkImpl from an optional path or use default NaiveNetwork
    pub fn from_file_or_default(path: Option<impl AsRef<Path>>) -> Self {
        match path {
            Some(path) => NetworkImpl::Regression(RegressionNetwork::from_file(path)),
            None => NetworkImpl::Naive(NaiveNetwork::new(10.0, 1.0)),
        }
    }
}

impl Network for NetworkImpl {
    fn measure(&self, collective: &Collective) -> Option<u64> {
        match self {
            NetworkImpl::Regression(n) => n.measure(collective),
            NetworkImpl::Naive(n) => n.measure(collective),
        }
    }
}
