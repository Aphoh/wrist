use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::PathBuf,
    process::exit,
};

use anyhow::{Context, Result};
use clap::Parser;
use wrist::graph::fx::parse_trace_result;

/// A simple program to parse and validate TraceResult protobuf messages from .bin files.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the directory containing .bin files.
    #[arg(short, long)]
    path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let path = PathBuf::from(args.path);

    if !path.is_dir() {
        eprintln!("Error: Path must be a directory.");
        exit(1);
    }

    let mut all_ok = true;

    for entry in fs::read_dir(path).context("Failed to read directory")? {
        let entry = entry.context("Failed to read directory entry")?;
        let file_path = entry.path();

        if file_path.is_file() && file_path.extension().map_or(false, |ext| ext == "bin") {
            println!("Parsing file: {}", file_path.display());

            let file = File::open(&file_path).context("Failed to open file")?;
            let mut reader = BufReader::new(file);

            match parse_trace_result(&mut reader) {
                Ok(_) => {
                    println!("  Successfully parsed.");
                }
                Err(e) => {
                    eprintln!("  Error parsing file: {:?}", e);
                    all_ok = false;
                }
            }
        }
    }

    if !all_ok {
        exit(1);
    }

    Ok(())
}