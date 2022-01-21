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
Performance results from SNN new norm

1. SNN-sims performance 

Simulations: from Moller and de Boissiere 2019 https://zenodo.org/record/3265189#.X85b1C1h3YV
Models: run using bash reproduce/1a.SNN_MB20_newnorm.sh

2. 14XDES performance

Simulations: run using Pippin Pippin/AM_DES5YR_SIMS.yml
Models: run using bash reproduce/1b.SNN_14XDES_newnorm.sh

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reproduce results paper SNN sims new norm"
    )

    parser.add_argument(
        "--path_models_SNN_MB20",
        default="./../snndump_MB20/models",
        type=str,
        help="Path to SNN models trained with MB20",
    )

    parser.add_argument(
        "--path_models_SNN_14XDES",
        default="./../snndump_14XDES/models",
        type=str,
        help="Path to SNN models trained with DESXL",
    )

    parser.add_argument(
        "--path_dump", default="./dump_DES5YR", type=str, help="Path to output"
    )
    args = parser.parse_args()

    # init
    os.makedirs(args.path_dump, exist_ok=True)
    path_plots = f"{args.path_dump}/plots/"
    os.makedirs(path_plots, exist_ok=True)

    #
    # Trained models with Moller & de Boissiere 2019/20 sims
    #

    # Stats
    metric_files_singleseed = du.get_metric_singleseed_files(
        args.path_models_SNN_MB20, "SNN_MB20"
    )
    du.get_stats_cal(metric_files_singleseed, args.path_dump, "MB20_norms")

    #
    # Trained models with 14XDES sims
    #

    # Default HP: model_name="vanilla_S_0_CLF_2_R_zspe_photometry_DF_1.0_N_*_lstm_32x2_0.05_128_True_mean_C"
    metric_files_singleseed = du.get_metric_singleseed_files(
        args.path_models_SNN_14XDES,
        "SNN_14XDES",
        model_name="vanilla_S_0_CLF_2_R_zspe_photometry_DF_1.0_N_*_lstm_32x2_0.05_128_True_mean_C",
    )
    du.get_stats_cal(metric_files_singleseed, args.path_dump, "14XDES_defaultHP")
