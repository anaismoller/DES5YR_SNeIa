import re, os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils import plot_utils as pu
from utils import data_utils as du
from utils import metric_utils as mu
from utils import logging_utils as lu

pd.options.mode.chained_assignment = None  # default='warn'


list_seeds = [0, 55, 100, 1000, 30469]

"""
Template changes using 14XDES simulations

Simulations: run using Pippin Pippin/AM_DES5YR_SIMS.yml

To reproduce models, run reproduce/1c.SNN_14XDES_templates.sh

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reproduce results paper SNN template changes 14XDES"
    )

    parser.add_argument(
        "--path_models",
        default="./../snndump_14XDES",
        type=str,
        help="Path to SNN output",
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
    path_models = args.path_models

    tmp = glob.glob(f"{path_models}/models_*")
    models = [m.split("_")[-1] for m in tmp]
    for mod in models:
        metric_files_singleseed = du.get_metric_singleseed_files(
            f"{path_models}/models_{mod}", f"SNN_14XDES_{mod}", model_name="vanilla*"
        )
        du.get_stats_cal(
            metric_files_singleseed, args.path_dump, f"14XDES_{mod}",
        )

    #
    # Additional analyses (not in paper, can be commented)
    #

    lu.print_blue("Load PREDS for validation")
    # get preds and SALT2 fits
    pred_dic = {}
    # to be used un the SALT2 fitted as well
    acc_dic = {}
    eff_dic = {}
    pur_dic = {}
    # loop over seeds
    for mod in ["14X"] + models:
        acc_dic[f"{mod}"] = []
        eff_dic[f"{mod}"] = []
        pur_dic[f"{mod}"] = []
        for seed in list_seeds:
            path_mod = f"models_{mod}" if mod != "14X" else "models"
            pred_file = glob.glob(
                f"{path_models}/{path_mod}/*S_{seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_32x2_0.05_128_True_mean_C/PRED*"
            )[0]
            pred_dic[f"{mod}_S_{seed}"] = du.load_pred(pred_file)
            (
                notbalancedaccuracy,
                accuracy,
                auc,
                purity,
                efficiency,
                truepositivefraction,
            ) = mu.performance_metrics(pred_dic[f"{mod}_S_{seed}"])
            acc_dic[f"{mod}"].append(accuracy)
            eff_dic[f"{mod}"].append(efficiency)
            pur_dic[f"{mod}"].append(purity)
        print(
            f"{mod}\n"
            f"Accuracy: {round(np.array(acc_dic[mod]).mean(),2)} \\pm {round(np.array(acc_dic[mod]).std(),2)} "
            f"Efficiency: {round(np.array(eff_dic[mod]).mean(),2)} \\pm {round(np.array(eff_dic[mod]).std(),2)} "
            f"Purity: {round(np.array(pur_dic[mod]).mean(),2)} \\pm {round(np.array(pur_dic[mod]).std(),2)} "
        )

    lu.print_blue("Load SALT2 fits and reduce to validation set")
    # SALT2 fitted lcs
    fits_dic = {}
    fits_dic["14X"] = du.load_fitfile(f"{args.path_fits}/PIP_AM_DES5YR_SIMS_14XDES/")
    for mod in models:
        tmp = du.load_fitfile(f"{args.path_fits}/PIP_AM_DES5YR_SIMS_14XDES_{mod}/")
        # validation set will be the same for all training seeds
        fits_dic[mod] = tmp[tmp.SNID.isin(pred_dic[f"{mod}_S_0"].SNID)]

    # 1. Get Accuracy for SALT2 fitted
    dic_pred_reduced = {}
    for mod in ["14X"] + models:
        acc_dic[f"{mod}"] = []
        eff_dic[f"{mod}"] = []
        pur_dic[f"{mod}"] = []
        for seed in list_seeds:
            # get preds that have a SALT2 fit
            pred_reduced = pd.merge(
                fits_dic[mod], pred_dic[f"{mod}_S_{seed}"], on="SNID", how="left"
            )
            # save for plot before ebalancing
            dic_pred_reduced[mod] = pred_reduced

            # rebalancing classes
            tmp2 = pred_reduced.groupby("target").count()["all_class0"]
            min_len_target = tmp2.idxmin()
            min_len_size = tmp2[min_len_target]
            # randomize
            targ0 = pred_reduced[pred_reduced["target"] == 0].reset_index()
            targ1 = pred_reduced[pred_reduced["target"] == 1].reset_index()
            idxs = np.random.randint(0, high=len(targ0), size=min_len_size)
            reduced_targ0 = targ0.loc[idxs]
            idxs = np.random.randint(0, high=len(targ1), size=min_len_size)
            reduced_targ1 = targ1.loc[idxs]
            pred_reduced_rebalanced = pd.concat([targ0, targ1])
            (
                notbalancedaccuracy,
                accuracy,
                auc,
                purity,
                efficiency,
                truepositivefraction,
            ) = mu.performance_metrics(pred_reduced_rebalanced)
            acc_dic[f"{mod}"].append(accuracy)
            eff_dic[f"{mod}"].append(efficiency)
            pur_dic[f"{mod}"].append(purity)

    lu.print_blue("Accuracy for only SALT2 fitted lcs")
    for k in acc_dic.keys():
        print(
            f"{k}\n"
            f"Accuracy: {round(np.array(acc_dic[k]).mean(),2)} \\pm {round(np.array(acc_dic[k]).std(),2)} "
            f"Efficiency: {round(np.array(eff_dic[k]).mean(),2)} \\pm {round(np.array(eff_dic[k]).std(),2)} "
            f"Purity: {round(np.array(pur_dic[k]).mean(),2)} \\pm {round(np.array(pur_dic[k]).std(),2)} "
        )

    # 2. Visualize with HD
    for k in dic_pred_reduced.keys():
        pu.plot_HD(
            dic_pred_reduced[k], path_output=f"{args.path_dump}/plots/{k}_HD.png"
        )
