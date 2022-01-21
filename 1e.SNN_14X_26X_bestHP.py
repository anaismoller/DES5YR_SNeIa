import re, os
import argparse
import pandas as pd
from pathlib import Path
from utils import conf_utils as cu
from utils import plot_utils as pu
from utils import data_utils as du
from utils import metric_utils as mu
from utils import logging_utils as lu
from utils import science_utils as su

pd.options.mode.chained_assignment = None  # default='warn'

"""
Best models using 14XDES and 26XBOOSTEDDES simulations

To reproduce models, run reproduce/1e.SNN_14XDES_bestHP and 1g.SNN_26XDESBOSSTED.sh

for PSNID validation 1e.SNN_14XDES_bestHP_val_PSNID.sh

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reproduce results paper SNN best 14X and 26X models"
    )

    parser.add_argument(
        "--path_models14X",
        default="./../snndump_14XDES",
        type=str,
        help="Path to 14X output",
    )

    parser.add_argument(
        "--path_models26X",
        default="./../snndump_26XBOOSTEDDES",
        type=str,
        help="Path to 26X output",
    )

    parser.add_argument(
        "--path_fits", default="./../2_LCFIT/", type=str, help="Path to output",
    )

    parser.add_argument(
        "--path_dump", default="./dump_DES5YR", type=str, help="Path to output"
    )
    args = parser.parse_args()

    # init
    os.makedirs(args.path_dump, exist_ok=True)
    path_plots = f"{args.path_dump}/plots/"
    os.makedirs(path_plots, exist_ok=True)

    # 14XDES
    metric_files_singleseed = du.get_metric_singleseed_files(
        f"{args.path_models14X}/models/",
        "SNN_14XDES_BESTHP/",
        model_name="vanilla_S_0_CLF_2_R_zspe_photometry_DF_1.0_N_*_lstm_64x4_0.05_512_True_mean",
    )
    du.get_stats_cal(metric_files_singleseed, args.path_dump, "14XDES_bestHP")

    # 26XBOOSTED
    metric_files_singleseed = du.get_metric_singleseed_files(
        f"{args.path_models26X}/models/",
        "SNN_26XBOOSTEDDES_BESTHP",
        model_name="vanilla_S_0_CLF_2_R_zspe_photometry_DF_1.0_N_*_lstm_64x4_0.05_1024_True_mean/",
    )
    du.get_stats_cal(metric_files_singleseed, args.path_dump, "26XBOOSTEDDES_bestHP")

    # Load SALT2 fits for later
    lu.print_blue("Load", "SALT2 fits 26XB")
    fits_26X = du.load_fitfile(f"{args.path_fits}/PIP_AM_DES5YR_SIMS_26XBOOSTEDDES/")

    # Detailed analysis: requires predictions
    lu.print_blue("Load", "PREDS 26XB")
    df_dic = {}
    df_txt_stats = pd.DataFrame(
        columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
    )
    for norm in ["cosmo", "cosmo_quantile"]:
        # Get predictions
        df_dic[norm] = du.get_preds_seeds_merge(
            cu.all_seeds,
            f"{args.path_models26X}/models/",
            norm=norm,
            model_suffix=f"_CLF_2_R_zspe_photometry_DF_1.0_N_{norm}_lstm_64x4_0.05_1024_True_mean",
        )
        # Ensemble methods
        df_dic, list_sets = du.add_ensemble_methods(df_dic, norm)

        for method, desc in cu.dic_sel_methods.items():
            list_seeds_sets = (
                cu.list_seeds_set[0] if method == "predicted_target_S_" else list_sets
            )
            # predicted_target_samepred_set
            df_txt_stats = mu.get_multiseed_performance_metrics(
                df_dic[norm],
                key_pred_targ_prefix=method,
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "balanced",
                    "method": desc,
                },
            )

    lu.print_green("26XBOOSTEDDES_bestHP with ensemble methods and extra stats")
    # print(df_txt_stats.to_string(index=False))#.to_latex())
    mu.reformatting_tolatex(df_txt_stats)

    lu.print_green(
        "Generalization with CC templates training:V19, validation: J17 or PSNID "
    )
    dic_hp = {"26X": "64x4_0.05_1024", "14X": "64x4_0.05_512"}

    for sim in ["14X", "26X"]:
        lu.print_green(f"Generalization with {sim} model")
        df_dic_generalization = {}

        # init for these sims
        model_hp = dic_hp[sim]
        list_s = cu.list_seeds_set[
            0
        ]  # just generalization one set cu.all_seeds if sim == "26X" else cu.list_seeds_set[0]
        sel_methods = (
            cu.dic_sel_methods.items()
            if sim == "26X"
            else {"predicted_target_S_": "single model"}.items()
        )

        df_txt_stats_generalization = pd.DataFrame(
            columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
        )
        for temp in ["PSNID", "J17"]:
            df_dic_generalization[temp] = {}

            for norm in ["cosmo", "cosmo_quantile"]:
                list_preds = []
                df_dic_generalization[temp][norm] = du.get_preds_seeds_merge(
                    list_s,
                    f"{args.path_models14X}/models_{temp}/",
                    norm=norm,
                    model_suffix=f"_CLF_2_R_zspe_photometry_DF_1.0_N_{norm}_lstm_{model_hp}_True_mean",
                )

                list_pred_targets = [
                    k
                    for k in df_dic_generalization[temp][norm].keys()
                    if "predicted_target" in k
                ]

                if sim == "26X":
                    # Ensemble methods
                    df_dic_generalization[temp], list_sets = du.add_ensemble_methods(
                        df_dic_generalization[temp], norm
                    )

                for method, desc in sel_methods:
                    list_seeds_sets = (
                        cu.list_seeds_set[0]
                        if method == "predicted_target_S_"
                        else list_sets
                    )

                    df_txt_stats_generalization = mu.get_multiseed_performance_metrics(
                        df_dic_generalization[temp][norm],
                        key_pred_targ_prefix=method,
                        list_seeds=list_seeds_sets,
                        df_txt=df_txt_stats_generalization,
                        dic_prefilled_keywords={
                            "norm": norm,
                            "dataset": temp,
                            "method": desc,
                        },
                    )

        # mu.reformatting_tolatex(df_txt_stats_generalization)
        print(df_txt_stats_generalization.to_string(index=False))

