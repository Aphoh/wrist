use std::{fs, io::BufReader, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wrist::{
    graph::{
        fx::{torch_titan::ParallelConfig, TraceBuilder},
        MissingProfiles,
    },
    kernels::KernelProfileImpl,
    network::NetworkImpl,
};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Sets the input file
    #[clap(short, long)]
    traces: PathBuf,
    #[clap(short, long)]
    network_profile: Option<PathBuf>,
    #[clap(short, long)]
    kernel_profile: Option<PathBuf>,
    #[clap(short, long, default_value_t = 80.0f32)]
    leaf_memory_gb: f32,
}

fn merge_missing_profiles(
    missing1: Option<MissingProfiles>,
    missing2: Option<MissingProfiles>,
) -> Option<MissingProfiles> {
    match (missing1, missing2) {
        (Some(mut missing1), Some(missing2)) => {
            missing1.merge(missing2);
            Some(missing1)
        }
        (Some(missing), None) | (None, Some(missing)) => Some(missing),
        (None, None) => None,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let kernel_profile = KernelProfileImpl::from_file_or_default(args.kernel_profile);
    let net = NetworkImpl::from_file_or_default(args.network_profile);

    let paths =
        glob::glob(args.traces.join("*.pb").to_str().unwrap())?.collect::<Result<Vec<_>, _>>()?;
    let graphs = paths
        .par_iter()
        .map(|path| {
            fs::File::open(path)
                .with_context(|| "Failed to open file")
                .map(BufReader::new)
                .and_then(|mut buf| TraceBuilder::parse(&mut buf))
        })
        .collect::<Result<Vec<TraceBuilder>>>()?;

    let mut hist = histo::Histogram::with_buckets(10);

    // Collect all results into a Vec
    let all_results: Vec<(ParallelConfig, u64, Option<MissingProfiles>)> = graphs
        .into_par_iter()
        .flat_map(|trace: TraceBuilder| {
            let dims = trace.parallel_dims().clone();
            match trace.build_compute_graph() {
                Ok((pc, graph)) => {
                    let time_us = graph.time(&net, &kernel_profile);
                    let res = match time_us {
                        Ok(time) => vec![(pc, time, None)],
                        Err(missing) => vec![(pc, u64::MAX, Some(missing))],
                    };
                    res
                }
                Err(e) => {
                    eprintln!("Failed to build compute graph for pc {:?}: {:?}", dims, e);
                    vec![]
                }
            }
        })
        .collect();

    // Add valid times to histogram
    for (_, time, _) in &all_results {
        if *time != u64::MAX {
            hist.add(*time);
        }
    }

    // Print histogram
    println!("Runtime distribution:\n{}", hist);

    // Find the best configuration
    let result =
        all_results
            .into_iter()
            .reduce(|(pc1, time1, missing1), (pc2, time2, missing2)| {
                let missing = merge_missing_profiles(missing1, missing2);
                if time1 < time2 {
                    (pc1, time1, missing)
                } else {
                    (pc2, time2, missing)
                }
            });

    if let Some((config, time, missing)) = result {
        if let Some(missing) = missing {
            println!("Missing profiles: {:?}", missing);
        }
        println!("Best trace: {:?}", config);
        println!("Time: {} us", time);
    } else {
        return Err(anyhow!("No valid traces found"));
    }
    Ok(())
}
