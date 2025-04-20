import argparse

import pandas as pd

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("input", type=str)
    args.add_argument("output", type=str)

    args = args.parse_args()

    matmuls = pd.read_csv(args.input)
    sample_agg = matmuls.groupby(["m", "n", "k"]).mean().drop(columns=["sample"])
    sample_agg["type"] = "matmul"
    sample_agg.to_csv(args.output)
