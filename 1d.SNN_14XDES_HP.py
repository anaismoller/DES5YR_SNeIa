import re, os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils import plot_utils as pu
from utils import data_utils as du
from utils import logging_utils as lu

"""
Hyper parameter search using 14XDES simulations

To reproduce models, run reproduce/1d.SNN_14XDES_HP.sh

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reproduce results paper SNN hyper parameter search DESXL"
    )

    parser.add_argument(
        "--path_models", default="./../snndump_14XDES", type=str, help="Path to output",
    )

    parser.add_argument(
        "--path_dump", default="./dump_DES5YR", type=str, help="Path to output"
    )
    args = parser.parse_args()

    # init
    os.makedirs(args.path_dump, exist_ok=True)
    path_plots = f"{args.path_dump}/plots/"
    os.makedirs(path_plots, exist_ok=True)

    metric_files = glob.glob(f"{args.path_models}/models/*S_0_*DF_0.2*/METRICS*")
    df = du.read_metric_files(metric_files)

    # Outputs
    # 1. accuracy file
    df.to_csv(
        f"{args.path_dump}/SNN_DESXL_HP_accuracy.csv",
        columns=["model_name", "all_accuracy"],
    )
    # 2. stats
    tmp_df = df[["model_name", "all_accuracy"]]
    # max
    idx_max = tmp_df.all_accuracy.argmax()
    max_acc = tmp_df.iloc[idx_max].values.tolist()
    # min value
    idx_min = tmp_df.all_accuracy.argmin()
    min_acc = tmp_df.iloc[idx_min].values.tolist()
    # mean and std
    mean_acc = round(tmp_df.all_accuracy.mean(), 2)
    std_acc = round(tmp_df.all_accuracy.std(), 2)
    # save in file
    text_file = open(f"{args.path_dump}/SNN_DESXL_HP_summary_stats.txt", "w")
    txt = f"max accuracy: {max_acc}"
    print(txt)
    text_file.write(f"{txt}\n")
    txt = f"mean accuracy: {mean_acc} \\pm {std_acc}"
    print(txt)
    text_file.write(f"{txt}\n")
    txt = f"delta accuracy max-min: {round(max_acc[-1]-min_acc[-1],1)}"
    print(txt)
    text_file.write(f"{txt}\n")
    text_file.close()

    lu.print_blue('Ranking')
    print(df[["reduced_model_name", "all_accuracy", "model_name"]].sort_values(
        by="all_accuracy"
    ))

    # save ranking
    df[["reduced_model_name", "all_accuracy", "model_name"]].sort_values(
        by="all_accuracy"
    ).to_csv(f"{args.path_dump}/SNN_DESXL_HP_ranking.csv")

