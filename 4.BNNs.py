import logging
import argparse
import ipdb
import numpy as np
import pandas as pd
import glob, os, sys, re
from pathlib import Path
from functools import reduce
from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn import metrics


from utils import cuts as cuts
from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import metric_utils as mu
from utils import science_utils as su
from utils import logging_utils as lu
from utils import utils_emcee_poisson as mc

plt.switch_backend("agg")
pd.options.mode.chained_assignment = None

"""
BNNs and real data
"""

nn_decode_dic = {"variational": "MC", "bayesian": "BBB"}


def get_ensemble_pred(path_preds="./PRED*set*.pickle"):
    ensemble_pred_files = glob.glob(path_preds)
    if len(ensemble_pred_files) < 1:
        lu.print_red("Need to run BNN_ensemble_uncertainties", path_preds)
    else:
        list_df = []
        for fname in ensemble_pred_files:
            this_set = Path(fname).stem.split("set")[-1]

            tmp = pd.read_pickle(fname)
            tmp = tmp.reset_index(drop=True)
            tmp["SNID"] = tmp["SNID"].astype(np.int32)

            tmp[f"predicted_target_average_probability_set_{this_set}"] = 1
            tmp.loc[
                tmp[f"all_class0_median"] > 0.5,
                f"predicted_target_average_probability_set_{this_set}",
            ] = 0
            tmp = tmp.rename(
                columns={
                    "all_class0_median": f"all_class0_median_set_{this_set}",
                    "all_class0_std": f"all_class0_std_set_{this_set}",
                }
            )
            if "target_std" in tmp.keys():
                tmp = tmp.drop(columns=["target_std"])
            list_df.append(tmp)
        # pred_ensemble = pd.concat(list_df)
        pred_ensemble = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=["SNID", "target"],), list_df
        )
        return pred_ensemble


def unc_average_behaviour(df, df_whole_distribution):
    tmp = []
    accuracy_list = []

    # single
    for seed in [
        k for k in cu.list_seeds_set[0] if f"all_class0_std_S_{k}" in df.keys()
    ]:
        # uncertainties
        tmp.append(df[f"all_class0_std_S_{seed}"].values)
        # accuracy
        df["pred_target"] = 1
        df.loc[df[f"all_class0_S_{seed}"] > 0.5, "pred_target"] = 0
        accuracy_list.append(
            100
            * metrics.balanced_accuracy_score(
                df["target"].astype(int), df["pred_target"].astype(int)
            )
        )
    print("single model")
    du.print_stats(accuracy_list, context=f"accuracies")
    du.print_stats(tmp, context=f"uncertainties")

    # ensemble approx
    tmp = []
    accuracy_list = []
    for k in [
        k for k in cu.list_sets if f"average_probability_set_{k}_meanstd" in df.keys()
    ]:
        # uncertainties
        tmp.append(df[f"average_probability_set_{k}_meanstd"].values)

        # accuracies
        df["pred_target"] = 1
        df.loc[df[f"average_probability_set_{k}"] > 0.5, "pred_target"] = 0
        accuracy_list.append(
            100
            * metrics.balanced_accuracy_score(
                df["target"].astype(int), df["pred_target"].astype(int)
            )
        )
    print("approx ensemble")
    du.print_stats(accuracy_list, context=f"accuracies")
    du.print_stats(tmp, context=f"uncertainties")

    # ensemble whole dist
    # accuracy
    for set in [
        k
        for k in cu.list_sets
        if f"predicted_target_set_{k}" in df_whole_distribution.keys()
    ]:
        accuracy_list = 100 * metrics.balanced_accuracy_score(
            df_whole_distribution["target"].astype(int),
            df_whole_distribution[f"predicted_target_set_{this_set}"].astype(int),
        )
    print("true ensemble")
    du.print_stats(accuracy_list, context=f"accuracies")
    du.print_stats(
        df_whole_distribution["all_class0_std_set_0"], context=f"uncertainties set 0"
    )


def plot_scatter_wcolor(ax, df, varx, vary, color, vmin, vmax):

    fig_tmp = ax.scatter(df[varx], df[vary], c=df[color], vmin=vmin, vmax=vmax)
    smooth_x = np.arange(df[varx].min(), df[varx].max(), 0.05)
    for thres in [0.2, 0.3]:
        tmp = len(df[df[vary] > thres])
        txt = r"$\sigma_P >$"
        ax.plot(
            smooth_x,
            thres * np.ones(len(smooth_x)),
            color="grey",
            linestyle="-." if thres == 0.2 else "--",
            label=f"{txt}{thres}: {tmp}",
        )
    ax.set_xlabel(varx)
    ax.set_ylabel("SNN uncertainty")
    ax.legend()
    return fig_tmp


def plot_uncertainties_cuts(
    preds_dic, preds_w_metadata_dic, photoIa_inpreds_dic, path_plots, suffix=""
):
    plt.clf()
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
    ax = gs.subplots(sharex=False, sharey=True)

    for i, nn in enumerate(["variational", "bayesian"]):
        preds = preds_dic[nn]
        preds_w_metadata = preds_w_metadata_dic[nn]
        photoIa_inpreds = photoIa_inpreds_dic[nn]

        n, bins, patches = ax[i].hist(
            preds["average_probability_set_0_meanstd"], label="DES 5-year candidates"
        )
        # Multi-season + detections
        df_metadata_w_dets = cuts.detections("DETECTIONS", preds_w_metadata, logger)
        preds_w_metadata_sel_multi = cuts.transient_status(
            "MULTI-SEASON", df_metadata_w_dets, logger
        )
        ax[i].hist(
            preds_w_metadata_sel_multi["average_probability_set_0_meanstd"],
            label="+ Multi-season cut",
        )

        # REDSHIFT
        preds_w_metadata_sel_z = cuts.redshift(
            "REDSHIFT", preds_w_metadata_sel_multi, logger
        )
        ax[i].hist(
            preds_w_metadata_sel_z["average_probability_set_0_meanstd"],
            label="+ Redshift cut",
        )

        # SALT converges
        cols_to_keep = ["SNID", "SNTYPE"] + [
            k for k in preds_w_metadata_sel_z.keys() if k not in data_fits.keys()
        ]
        preds_w_metadata_sel_SALT = cuts.salt_basic(
            "SALT2 loose selection cuts",
            preds_w_metadata_sel_z[cols_to_keep],
            data_fits,
            logger,
        )
        ax[i].hist(
            preds_w_metadata_sel_SALT["average_probability_set_0_meanstd"],
            label="+ SALT2 loose selection cuts",
        )
        # PRIMUS
        hostz_info = pd.read_csv("extra_lists/SNGALS_DLR_RANK1_INFO.csv")
        hostz_info["SPECZ_CATALOG"] = hostz_info.SPECZ_CATALOG.apply(
            lambda x: x[2:-1].strip(" ")
        )
        SNID_to_keep = hostz_info[hostz_info.SPECZ_CATALOG != "PRIMUS"].SNID.to_list()
        preds_w_metadata_sel_primus = preds_w_metadata_sel_SALT[
            preds_w_metadata_sel_SALT.SNID.isin(SNID_to_keep)
        ]
        # JLA
        preds_w_metadata_sel_JLA = su.apply_JLA_cut(preds_w_metadata_sel_primus)
        ax[i].hist(
            preds_w_metadata_sel_JLA["average_probability_set_0_meanstd"],
            label="+ JLA-like cuts",
        )
        ax[i].hist(
            photoIa_inpreds["average_probability_set_0_meanstd"],
            label="Baseline DES JLA (RNN ensemble P>0.5)",
            bins=bins,
        )
        ax[i].hist(
            preds_w_metadata[preds_w_metadata["REDSHIFT_FINAL"] < 0][
                "average_probability_set_0_meanstd"
            ],
            label="DES 5-year candidates without redshift",
            bins=bins,
            histtype="step",
            edgecolor="black",
            linewidth=3,
        )
        no_z_high_p = preds_w_metadata[
            (preds_w_metadata["REDSHIFT_FINAL"] < 0)
            & (preds_w_metadata["average_probability_set_0"] > 0.5)
        ]
        print(
            f"# events without redshift and {nn_decode_dic[nn]} prob>0.5: {len(no_z_high_p)}"
        )

        ax[i].set_yscale("log")
        lab = f"{nn_decode_dic[nn]} classification uncertainty"
        ax[i].set_xlabel(lab, fontsize=16)
    ax[0].set_ylabel("# events", fontsize=16)
    # ax[i].legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0,fontsize=16)
    ax[i].legend(fontsize=16)

    plt.savefig(f"{path_plots}/uncertainties_cuts_{suffix}.png")


def setup_logging():
    logger = None

    # Create logger using python logging module
    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.DEBUG)

    logging_handler_err = logging.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(logging.WARNING)

    logger = logging.getLogger("localLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging_handler_out)
    logger.addHandler(logging_handler_err)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("logpaper.log", mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


if __name__ == "__main__":

    DES5yr = os.getenv("DES5yr")
    DES = os.getenv("DES")

    parser = argparse.ArgumentParser(description="Code to reproduce results paper")

    parser.add_argument(
        "--path_data_fits",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_JLA_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits",
    )
    parser.add_argument(
        "--path_data_fits_redshiftsaltfitted",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_FITZ_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits",
    )
    parser.add_argument(
        "--path_data_class",
        type=str,
        default=f"{DES}/DES5YR/data_preds/",
        help="Path to data predictions",
    )
    parser.add_argument(
        "--path_dump",
        default=f"./dump_DES5YR",
        type=str,
        help="Path to output & sample",
    )
    parser.add_argument(
        "--path_data",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits",
        help="Path to data",
    )
    parser.add_argument(
        "--path_models26X",
        default="./../snndump_26XBOOSTEDDES",
        type=str,
        help="Path to 26X output",
    )
    parser.add_argument(
        "--path_models14X",
        default="./../snndump_14XDES",
        type=str,
        help="Path to 14X output",
    )
    parser.add_argument(
        "--path_sim_fits",
        type=str,
        default=f"{DES}/DES5YR/snndump_5X/2_LCFIT/JLA_5XDES/output/PIP_AM_DES5YR_SIMS_TEST_5XDES/FITOPT000.FITRES.gz",
        help="Path to simualtion SALT2 fits (for selection efficiency)",
    )
    parser.add_argument(
        "--path_sim_class",
        type=str,
        default=f"{DES}/DES5YR/snndump_5X/models/",
        help="Path to sim predictions",
    )

    # Init
    logger = setup_logging()
    args = parser.parse_args()
    path_data_fits = args.path_data_fits
    path_data_class = args.path_data_class
    path_dump = args.path_dump
    path_data = args.path_data
    path_models26X = args.path_models26X
    path_sim_fits = args.path_sim_fits
    path_sim_class = args.path_sim_class

    path_samples = f"{path_dump}/samples"
    norm = "cosmo_quantile"

    #
    # SIMULATIONS
    #
    print("")
    lu.print_green("SIMULATIONS")

    # Load 5X simulation fits
    lu.print_blue("Loading sim fits", path_sim_fits)
    sim_fits = du.load_salt_fits(path_sim_fits)
    # same redshift range than data
    sim_fits_zrange = sim_fits[(sim_fits["zHD"] > 0.05) & (sim_fits["zHD"] < 1.3)]
    # now options
    sim_JLA_fits = su.apply_JLA_cut(sim_fits_zrange)
    sim_Ia_fits = sim_fits_zrange[sim_fits_zrange.SNTYPE == 101]
    sim_Ia_JLA_fits = su.apply_JLA_cut(sim_Ia_fits)

    # For approximate ensemble methods
    lu.print_blue("Load", "PREDS 26XB")
    df_dic = {}
    for nn in ["variational", "bayesian"]:
        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"
        os.makedirs(path_plots, exist_ok=True)
        lu.print_green(f"{nn_decode_dic[nn]}")
        df_dic[nn] = {}
        df_txt_stats = pd.DataFrame(
            columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
        )
        # aproximation
        df_dic[nn] = du.load_merge_all_preds(
            path_class=f"{args.path_models26X}/models/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )

        # stats
        for method, desc in cu.dic_sel_methods.items():
            list_seeds_sets = (
                cu.list_seeds_set[0]
                if method == "predicted_target_S_"
                else cu.list_sets
            )
            # predicted_target_samepred_set
            df_txt_stats = mu.get_multiseed_performance_metrics(
                df_dic[nn][norm],
                key_pred_targ_prefix=method,
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "balanced",
                    "method": desc,
                },
            )

        print(mu.reformatting_tolatex(df_txt_stats, norm_list=["cosmo_quantile"]))

    # Evaluating the approximation
    # mean complete distribution vs. mean average probabilities
    #
    lu.print_blue("True vs. Approximation probabilities and uncertainties")
    pred_ensemble_26XB_dic = {}
    for nn in ["variational", "bayesian"]:
        # real ensemble
        pred_ensemble_26XB_dic[nn] = get_ensemble_pred(
            f"{path_models26X}/models/PRED_{nn}*set*.pickle"
        )
        # approx ensemble
        pred_approx = df_dic[nn]["cosmo_quantile"][
            [
                k
                for k in [
                    "SNID",
                    "average_probability_set_0",
                    "average_probability_set_0_meanstd",
                    "average_probability_set_0_stdprob",
                    "average_probability_set_1",
                    "average_probability_set_1_meanstd",
                    "average_probability_set_1_stdprob",
                    "average_probability_set_2",
                    "average_probability_set_2_meanstd",
                    "average_probability_set_2a_stdprob",
                ]
                if k in df_dic[nn]["cosmo_quantile"].keys()
            ]
        ]

        merged = pd.merge(pred_ensemble_26XB_dic[nn], pred_approx, on="SNID")

        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"
        os.makedirs(path_plots, exist_ok=True)

        diff_dic = {
            "probabilities": merged["all_class0_median_set_0"]
            - merged["average_probability_set_0"],
            "meanuncertainty": merged["all_class0_std_set_0"]
            - merged["average_probability_set_0_meanstd"],
            "stdprobability_asuncertainty": merged["all_class0_std_set_0"]
            - merged["average_probability_set_0_stdprob"],
        }

        for k, v in diff_dic.items():
            fig = plt.figure()
            n, _, _ = plt.hist(v)
            plt.plot([0, 0], [0, max(n)], linestyle="--", color="grey")
            plt.plot(
                [np.mean(v), np.mean(v)], [0, max(n)], linestyle="--", color="orange",
            )
            plt.yscale("log")
            plt.xlabel(f"delta {k}")
            info_str = f"{np.mean(v):.2f} pm {np.std(v):.2f}"
            plt.title(info_str)
            plt.savefig(f"{path_plots}/hist_approx_{k}.png")
            print(f"{nn_decode_dic[nn]} {k} {info_str}")

            fig = plt.figure()
            y_axis = (
                merged["all_class0_median_set_0"]
                if k == "probabilities"
                else merged["all_class0_std_set_0"]
            )
            plt.scatter(v, y_axis)
            plt.xlabel(f"delta {k}")
            plt.ylabel(k)
            plt.savefig(f"{path_plots}/scatter_approx_{k}.png")

        # prob vs uncertainty
        del fig
        fig = plt.figure()
        plt.scatter(
            pred_ensemble_26XB_dic[nn]["all_class0_median_set_0"],
            pred_ensemble_26XB_dic[nn]["all_class0_std_set_0"],
        )
        plt.xlabel("mean prob")
        plt.xlabel("std prob")
        plt.savefig(f"{path_plots}/scatter_probvssigma_sim.png")
        del fig

    # Metrics in 5XDES
    # USING SELECTION AND JLA-CUTS
    lu.print_blue("Load", "PREDS 5X")
    df_dic_5X = {}
    df_dic_5X_wfits_JLA = {}
    for nn in ["variational", "bayesian"]:
        lu.print_green(f"{nn_decode_dic[nn]}")
        df_dic_5X[nn] = {}
        df_txt_stats_selcuts = pd.DataFrame(
            columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
        )
        df_txt_stats_JLA = pd.DataFrame(
            columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
        )
        # aproximation
        df_dic_5X[nn] = du.load_merge_all_preds(
            path_class=f"{args.path_sim_class}/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )
        # add sel cuts + JLA cuts
        df_dic_5X_wfits = pd.merge(sim_fits, df_dic_5X[nn]["cosmo_quantile"])
        df_dic_5X_wfits_JLA[nn] = su.apply_JLA_cut(df_dic_5X_wfits)

        # stats
        for method, desc in cu.dic_sel_methods.items():
            list_seeds_sets = (
                cu.list_seeds_set[0]
                if method == "predicted_target_S_"
                else cu.list_sets
            )
            df_txt_stats_JLA = mu.get_multiseed_performance_metrics(
                df_dic_5X_wfits_JLA[nn],
                key_pred_targ_prefix=method,
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats_JLA,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "5XDES JLAcuts",
                    "method": desc,
                },
            )

        print(mu.reformatting_tolatex(df_txt_stats, norm_list=["cosmo_quantile"]))
    # #
    # DATA
    #
    print("")
    lu.print_green("DATA")

    #
    # Load sample with baseline RNN and its fits
    #
    path_sample = f"{path_samples}/photoIa_cosmo_quantile_average_probability_set_0.csv"
    lu.print_blue("Loading sample & fits set 0", path_sample)
    sample_baseline = pd.read_csv(path_sample)
    data_fits = du.load_salt_fits(path_data_fits)
    sample_baseline_fits = data_fits[data_fits.SNID.isin(sample_baseline.SNID)]
    lu.print_green(
        f"Previous photo Ia sample baseline {len(sample_baseline)} (baseline ensemble set 0)"
    )
    # S_0
    path_sample = f"{path_samples}/photoIa_cosmo_quantile_S_0.csv"
    lu.print_blue("Loading sample & fits S 0", path_sample)
    sample_baseline_S_0 = pd.read_csv(path_sample)
    sample_baseline_fits_S_0 = data_fits[data_fits.SNID.isin(sample_baseline_S_0.SNID)]

    # Load BNN PREDICTIONS on data
    data_preds = {}
    preds = {}
    preds_for_RNN_sample = {}
    preds_for_RNN_sample_ensemble = {}
    pred_ensemble_data = {}
    for nn in ["variational", "bayesian"]:
        lu.print_blue(f"Loading predictions on data {nn_decode_dic[nn]}")
        data_preds[nn] = du.load_merge_all_preds(
            path_class=f"{path_data_class}/snndump_26XBOOSTEDDES/models/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )
        preds_tmp = data_preds[nn]["cosmo_quantile"]

        # merge with salt
        preds[nn] = pd.merge(preds_tmp, data_fits, on="SNID", how="left")

        # compute ensemble uncertainties
        pred_ensemble_data[nn] = get_ensemble_pred(
            f"{path_data_class}/snndump_26XBOOSTEDDES/models/PRED_{nn}*set*.pickle"
        )

        # Get stats of uncertainties for DES 5-year sample
        preds_for_RNN_sample[nn] = preds[nn][
            preds[nn].SNID.isin(sample_baseline_fits.SNID.values)
        ]

    #
    # Without selection cuts
    #
    lu.print_green("Uncertainties correlations with parameters")
    lu.print_green("No selection cuts, using single model")
    list_vars_to_plot = [
        "all_class0_S_0",
        "SNRMAX3",
        "zHD",
    ]

    path_plots = f"{path_dump}/plots_sample_bothBNNs/"
    os.makedirs(path_plots, exist_ok=True)

    def plot_uncertainties_mosaic(
        var="all_class0_S_0",
        var_unc="all_class0_std_S_0",
        suffix="S_0",
        plot_probs=False,
        list_vars_to_plot=["all_class0_S_0", "SNRMAX3", "zHD"],
    ):
        counter = 0
        plt.clf()
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, len(list_vars_to_plot), hspace=0, wspace=0.05)
        axs = gs.subplots(sharex="col", sharey=True)
        tmp_bins = {
            "zHD": np.linspace(0.05, 1.2, 8),
            "c": np.linspace(-0.4, 0.4, 8),
            "x1": np.linspace(-4, 4, 8),
            "all_class0_S_0": np.linspace(0, 1, 8),
            "SNRMAX3": np.linspace(3, 20, 8),
            "m0obs_i": np.linspace(21, 25, 8),
        }
        for nn in ["variational", "bayesian"]:
            for i, k in enumerate(list_vars_to_plot):
                if (
                    k == "SNRMAX3"
                ):  # signal-to-noise of the 3rd brightest (3rd highest S/N)
                    # only plotting usual SN range..
                    # <0 handful events, template issues
                    # >100 handful events
                    to_plot = preds[nn][
                        (preds[nn].SNRMAX3 > 3) & (preds[nn].SNRMAX3 < 20)
                    ]
                else:
                    to_plot = preds[nn]
                # add mean uncertainty per bin
                mean_bins = tmp_bins[k][:-1] + (tmp_bins[k][1] - tmp_bins[k][0]) / 2
                to_plot[f"{k}_bin"] = pd.cut(
                    to_plot.loc[:, (k)], tmp_bins[k], labels=mean_bins
                )
                tmp = to_plot.groupby(f"{k}_bin").median()
                # uncertainty in data
                result_low = (
                    to_plot[[f"{k}_bin", var_unc]].groupby(f"{k}_bin").quantile(0.16)
                )[var_unc].values
                result_high = (
                    to_plot[[f"{k}_bin", var_unc]].groupby(f"{k}_bin").quantile(0.84)
                )[var_unc].values
                axs[counter][i].fill_between(
                    mean_bins,
                    result_low,
                    result_high,
                    color="maroon",
                    alpha=0.2,
                    zorder=90,
                )
                if plot_probs:
                    sc = axs[counter][i].scatter(
                        mean_bins,
                        tmp[var_unc].values,
                        c=tmp[var].values,
                        s=80,
                        marker="D",
                        edgecolors="black",
                        linewidth=3,
                        zorder=100,
                        cmap="cividis",
                        vmin=0,
                        vmax=1,
                    )
                else:
                    axs[counter][i].errorbar(
                        mean_bins, tmp[var_unc].values, color="maroon"
                    )
                # uncertainty in sim
                sim_to_plot = pd.merge(
                    sim_fits, df_dic[nn]["cosmo_quantile"], on="SNID"
                )
                sim_to_plot[f"{k}_bin"] = pd.cut(
                    sim_to_plot.loc[:, (k)], tmp_bins[k], labels=mean_bins
                )
                tmp_sim = sim_to_plot.groupby(f"{k}_bin").median()
                tmp_sim_err = sim_to_plot.groupby(f"{k}_bin").std()[var_unc].values
                # uncertainty insim
                result_low = (
                    sim_to_plot[[f"{k}_bin", var_unc]]
                    .groupby(f"{k}_bin")
                    .quantile(0.16)
                )[var_unc].values
                result_high = (
                    sim_to_plot[[f"{k}_bin", var_unc]]
                    .groupby(f"{k}_bin")
                    .quantile(0.84)
                )[var_unc].values
                axs[counter][i].fill_between(
                    mean_bins,
                    result_low,
                    result_high,
                    color="darkgrey",
                    alpha=0.8,
                    zorder=-20,
                )
                axs[counter][i].errorbar(
                    mean_bins,
                    tmp_sim[var_unc].values,
                    color="black",
                    alpha=0.5,
                    linestyle="--",
                )

                axs[counter][i].set_yscale("log")
                axs[counter][i].set_ylim(10e-6, 0.4)

                if k == "all_class0_S_0":
                    xlab = "classification probability"
                elif k == "m0obs_i":
                    xlab = r"$i_{peak}$"
                else:
                    xlab = k
                axs[1][i].set_xlabel(xlab)
            counter += 1
        if plot_probs:
            clb = plt.colorbar(sc, location="bottom", ax=axs, fraction=0.1)
            clb.ax.set_title("classification probability", fontsize=16)
            x_axis_title = 2
        else:
            x_axis_title = 1
        axs[0][x_axis_title].set_title("MC dropout", fontsize=18)
        axs[1][x_axis_title].set_title("Bayes by Backprop", fontsize=18)
        axs[0][0].set_ylabel("classification uncertainty", fontsize=16)
        axs[1][0].set_ylabel("classification uncertainty", fontsize=16)
        plt.savefig(f"{path_plots}/scatter_uncertainty_mosaic_{suffix}.png")
        plt.clf()

    plot_uncertainties_mosaic(
        var="all_class0_S_0", var_unc="all_class0_std_S_0", suffix="S_0"
    )
    plot_uncertainties_mosaic(
        var="average_probability_set_0",
        var_unc="average_probability_set_0_meanstd",
        suffix="set_0",
    )
    plot_uncertainties_mosaic(
        var="average_probability_set_0",
        var_unc="average_probability_set_0_meanstd",
        plot_probs=True,
        suffix="set_0_wprobs",
        list_vars_to_plot=["SNRMAX3", "zHD", "c", "x1", "m0obs_i"],
    )

    #
    # Sample analysis
    #

    # selection cuts + JLA
    lu.print_blue("Loading data with only selection cuts")
    path_sample_selcuts = f"{path_samples}/only_selection_cuts.csv"
    tmp_sample_selcuts_only = pd.read_csv(path_sample_selcuts)
    selcuts_fits = data_fits[data_fits.SNID.isin(tmp_sample_selcuts_only.SNID.values)]
    selcuts_JLA = su.apply_JLA_cut(selcuts_fits)
    lu.print_blue(f"Selecting data that pass selection cuts + JLA", len(selcuts_JLA))

    # Samples with selection and JLA cuts
    preds_selcuts_JLA = {}
    sample_S_0 = {}
    sample_set_0_beforeJLA = {}
    sample_set_0 = {}
    sample_set_0_sigma = {}
    for nn in ["variational", "bayesian"]:
        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"

        #
        # sample
        preds_selcuts_JLA[nn] = preds[nn][
            (preds[nn].SNID.isin(selcuts_JLA.SNID.values))
        ]
        # photoIa sample with single seed
        photoIa_S_0 = cuts.photo_sel_target(
            preds_selcuts_JLA[nn], target_key=f"predicted_target_S_0", df_out=True,
        )

        df_photoIa_stats_singleapproxens = du.get_sample_stats(preds_selcuts_JLA[nn])

        #
        # add a sample with uncertainty threshold
        thres_std = np.round(preds[nn]["all_class0_std_S_0"].quantile(0.99), 3)
        # single seed
        unc_cut = preds_selcuts_JLA[nn]["all_class0_std_S_0"] < thres_std
        df_photoIa_stats_singleapprox_plussigma = du.get_sample_stats(
            preds_selcuts_JLA[nn][unc_cut],
            suffix="_JLA_sigmacut",
            methods=["single_model"],
        )
        # ensemble (using approximation)
        unc_cut = preds_selcuts_JLA[nn]["average_probability_set_0_meanstd"] < thres_std
        df_photoIa_stats_ensemble_plussigma = du.get_sample_stats(
            preds_selcuts_JLA[nn][unc_cut],
            suffix="_JLA_sigmacut",
            methods=["average_probability"],
        )
        df_photoIa_stats_singleapproxens_plussigma = pd.concat(
            [
                df_photoIa_stats_singleapprox_plussigma,
                df_photoIa_stats_ensemble_plussigma,
            ]
        )
        # merge stats
        df_stats = pd.merge(
            df_photoIa_stats_singleapproxens,
            df_photoIa_stats_singleapproxens_plussigma,
            on="method",
        )
        # add the photoIa sample with ensemble
        # need to do this manually due to format
        photoIa_avg_prob_set_0 = cuts.photo_sel_target(
            preds_selcuts_JLA[nn],
            target_key=f"predicted_target_average_probability_set_0",
            df_out=True,
        )
        photoIa_avg_prob_set_0_sigmacut = cuts.photo_sel_target(
            preds_selcuts_JLA[nn][unc_cut],
            target_key=f"predicted_target_average_probability_set_0",
            df_out=True,
        )
        dic_tmp = {
            "norm": norm,
            "method": "photo Ia no z av.prob. set 0",
            "photoIa": len(photoIa_avg_prob_set_0[0]),
            "specIa": len(photoIa_avg_prob_set_0[1]),
            "specCC": len(photoIa_avg_prob_set_0[2]),
            "specOther": len(photoIa_avg_prob_set_0[3]),
            "photoIa_JLA_sigmacut": len(photoIa_avg_prob_set_0_sigmacut[0]),
            "specIa_JLA_sigmacut": len(photoIa_avg_prob_set_0_sigmacut[1]),
            "specCC_JLA_sigmacut": len(photoIa_avg_prob_set_0_sigmacut[2]),
            "specOther_JLA_sigmacut": len(photoIa_avg_prob_set_0_sigmacut[3]),
        }
        # join true vs approx
        df_stats = df_stats.append(dic_tmp, ignore_index=True)

        latex_table = df_stats.to_latex(
            index=False,
            columns=[
                "method",
                "photoIa",
                "specIa",
                "photoIa_JLA_sigmacut",
                "specIa_JLA_sigmacut",
            ],
        )
        lu.print_blue(
            f"{nn_decode_dic[nn]} using approximation for both prob and uncertainties"
        )
        print(latex_table)

        # overlap between all these alternatives
        dic_venn = {
            "DES-5yr": set(sample_baseline_fits.SNID.values),
            f"{nn_decode_dic[nn]} single model S 0": set(photoIa_S_0[0].SNID.values),
            f"{nn_decode_dic[nn]} ensemble set 0": set(
                photoIa_avg_prob_set_0[0].SNID.values
            ),
        }
        pu.plot_venn(dic_venn, path_plots=path_plots, suffix=nn)

        # save samples
        sample_S_0[nn] = photoIa_S_0[0]
        sample_set_0[nn] = photoIa_avg_prob_set_0[0]
        sample_set_0_sigma[nn] = photoIa_avg_prob_set_0_sigmacut[0]
        # extra smaple before JLA
        tmp_beforeJLA = preds[nn][(preds[nn].SNID.isin(selcuts_fits.SNID.values))]
        sample_set_0_beforeJLA[nn] = cuts.photo_sel_target(
            tmp_beforeJLA,
            target_key=f"predicted_target_average_probability_set_0",
            df_out=True,
        )[0]

        # plot to see possible biases for c,x1,z
        list_df = [
            sample_S_0[nn],
            sample_set_0[nn],
            sample_set_0_sigma[nn],
        ]
        list_labels = [
            "photo Ia S 0 (BNN)",
            "photo Ia set 0 (BNN)",
            "photo Ia set 0+unc (BNN)",
        ]
        nameout = f"{path_plots}/hists_{nn_decode_dic[nn]}_photoIa_samples.png"
        plt.clf()
        plt.close("all")
        pu.plot_cx1z_histos(
            list_df + [sample_baseline_fits], list_labels + ["photo Ia (RNN)"], nameout
        )
        plt.clf()
        plt.close("all")

        # missing & new events
        missing_fromsample = {}
        new_BNN = {}
        for i, df in enumerate(list_df):
            idxs = df.SNID.values
            missing_SNID_fromsample = [
                k for k in sample_baseline.SNID.values if k not in idxs
            ]
            lu.print_green(f"{list_labels[i]} {nn_decode_dic[nn]}")
            print(f"Missing RNN DES 5-year in sample {len(missing_SNID_fromsample)}",)
            missing_fromsample[list_labels[i]] = sample_baseline[
                sample_baseline.SNID.isin(missing_SNID_fromsample)
            ]
            new_SNID_inBNN = [
                k for k in df.SNID.values if k not in sample_baseline.SNID.values
            ]
            print(
                f"New BNN not in RNN DES 5-year in sample {len(new_SNID_inBNN)} ({round(100*len(new_SNID_inBNN)/len(sample_baseline),1)}%)",
            )
            new_BNN[list_labels[i]] = df[df.SNID.isin(new_SNID_inBNN)]

        nameout = f"{path_plots}/hists_{nn_decode_dic[nn]}_new.png"
        list_labels = [f"new in BNN {k}" for k in new_BNN.keys()]
        pu.plot_cx1z_histos(
            [new_BNN[k] for k in new_BNN.keys()], list_labels, nameout,
        )

        nameout = f"{path_plots}/hists_{nn_decode_dic[nn]}_missing.png"
        list_labels = [f"missing DES5yr in BNN {k}" for k in missing_fromsample.keys()]
        pu.plot_cx1z_histos(
            [missing_fromsample[k] for k in missing_fromsample.keys()],
            list_labels,
            nameout,
        )

        logger.info("")
        logger.info(f"SAMPLE CONTAMINATION {nn_decode_dic[nn]}")
        # hack to use the same contaminant inspection
        dic_photoIa_sel = {
            "S_0": sample_S_0[nn],
            "average_probability_set_0": sample_set_0[nn],
        }
        dic_tag_SNIDs = cuts.stats_possible_contaminants(
            dic_photoIa_sel, method_list=["S", "average_probability_set"]
        )

        logger.info("")
        logger.info("SAMPLE PROPERTIES")

        # For: DES 5-year sample, MC selected no sigma, MC selected with sigma, DES 5-year sample + sigma
        list_df = [
            sample_baseline_fits,
            sample_set_0[nn],
            sim_Ia_JLA_fits,
        ]
        list_labels = [
            "photo Ia JLA (RNN)",
            f"{nn_decode_dic[nn]} ensemble set 0",
            "sim Ia JLA",
        ]
        pu.overplot_salt_distributions_lists(
            list_df, list_labels=list_labels, path_plots=path_plots, suffix=nn
        )

        # add m0obs_i
        pu.plot_mosaic_histograms_listdf(
            [sim_Ia_JLA_fits, sample_set_0[nn], sample_set_0_sigma[nn]],
            list_labels=[
                "sim Ia JLA",
                "ensemble set 0",
                "single model + uncertainty cut",
            ],
            path_plots=path_plots,
            suffix=nn,
            list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        )

    #
    # Both BNNs
    #
    lu.print_green("Samples both BNNs")
    path_plots = f"{path_dump}/plots_sample_bothBNNs/"
    os.makedirs(path_plots, exist_ok=True)

    # Both BNNs sample properties
    list_df = [
        sample_baseline_fits,
        sample_set_0["variational"],
        sample_set_0["bayesian"],
        sim_Ia_JLA_fits,
    ]
    list_labels = [
        "photo Ia JLA (RNN)",
        f"photo Ia JLA ({nn_decode_dic['variational']}) prob. av. set 0",
        f"photo Ia JLA ({nn_decode_dic['bayesian']}) prob. av. set 0",
        "sim Ia JLA",
    ]
    pu.overplot_salt_distributions_lists(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        data_color_override=True,
    )

    # see overlap between all these alternatives
    dic_venn = {
        "photo Ia JLA (RNN)": set(sample_baseline_fits.SNID.values),
        f"photo Ia JLA ({nn_decode_dic['variational']})": set(
            sample_set_0["variational"].SNID.values
        ),
        f"photo Ia JLA ({nn_decode_dic['bayesian']})": set(
            sample_set_0["bayesian"].SNID.values
        ),
    }
    pu.plot_venn(dic_venn, path_plots=path_plots, suffix="ensembleRNNBNNs")

    #
    # Uncertainties contribution
    #

    # using approximation only
    fig = plt.figure(figsize=(8, 5))
    for i, nn in enumerate(["variational", "bayesian"]):

        # sim
        tmp_sim = df_dic[nn]["cosmo_quantile"]
        tmp_sim_photoIa = df_dic[nn]["cosmo_quantile"][
            df_dic[nn]["cosmo_quantile"]["average_probability_set_0"] > 0.5
        ]
        tmp_sim_photoIa_JLA = tmp_sim_photoIa[
            tmp_sim_photoIa.SNID.isin(sim_Ia_JLA_fits.SNID.values)
        ]
        list_to_plot = [
            tmp_sim,
            tmp_sim_photoIa_JLA,
        ]
        to_plot_median = [
            k["average_probability_set_0_meanstd"].median() for k in list_to_plot
        ]
        low = [
            k["average_probability_set_0_meanstd"].quantile(0.16) for k in list_to_plot
        ]
        high = [
            k["average_probability_set_0_meanstd"].quantile(0.84) for k in list_to_plot
        ]
        to_plot_low = np.array(to_plot_median) - np.array(low)
        to_plot_high = np.array(high) - np.array(to_plot_median)
        plt.errorbar(
            np.arange(len(to_plot_median)) + (i * 0.3),
            to_plot_median,
            yerr=[to_plot_low, to_plot_high],
            fmt="s",
            color=pu.ALL_COLORS_nodata[i],
            elinewidth=10,
            ms=20,
            zorder=-10,
            alpha=0.4,
        )
        # data
        list_to_plot = [
            preds[nn]["average_probability_set_0_meanstd"],  # allDES no sel cuts
            sample_set_0[nn]["average_probability_set_0_meanstd"],  # BNN photo Ia JLA
        ]
        to_plot_median = [k.median() for k in list_to_plot]
        low = [k.quantile(0.16) for k in list_to_plot]
        high = [k.quantile(0.84) for k in list_to_plot]
        to_plot_low = np.array(to_plot_median) - np.array(low)
        to_plot_high = np.array(high) - np.array(to_plot_median)
        plt.errorbar(
            np.arange(len(to_plot_median)) + (i * 0.3),
            to_plot_median,
            yerr=[to_plot_low, to_plot_high],
            fmt="o",
            ms=10,
            label=f"{nn_decode_dic[nn]} BNN",
            color=pu.ALL_COLORS_nodata[i],
        )
    plt.legend()
    plt.ylabel("classification uncertainty")
    plt.xticks(
        ticks=np.arange(len(to_plot_median)) + 0.12,
        labels=[
            "DES 5-year candidates",
            # "photo Ia (BNN)",
            # "Baseline DES JLA",
            "Baseline BNN JLA",
        ],
    )
    plt.grid(True, which="both", axis="y")
    plt.savefig(f"{path_plots}uncertainties_allDES_samples.png")
    plt.clf()

    print("")
    lu.print_green("FROM DES TO RUBIN")
    # TOWARDS RUBIN
    spec = data_fits[
        (data_fits.SNTYPE.isin(cu.spec_tags["Ia"]))  # & (data_fits.zHD > 0.2)
    ]
    spec_JLA = su.apply_JLA_cut(spec)
    list_df = [
        spec_JLA,
        sample_baseline_fits,
        sample_set_0["variational"],  # BNN photo Ia JLA
        sample_set_0["bayesian"],  # BNN photo Ia JLA
    ]
    list_labels = ["spec Ia", "Baseline DES JLA", "Baseline MC JLA", "Baseline BBB JLA"]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="photo_spec",
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        data_color_override=True,
        chi_bins=False,
    )

    print("")
    lu.print_green("Uncertainties & representativity")
    lu.print_blue("Load", "PREDS for J17 and PSNID using 26XB")
    df_dic_J17 = {}
    df_dic_PSNID = {}
    pred_ensemble_PSNID_dic = {}
    pred_ensemble_J17_dic = {}
    for nn in ["variational", "bayesian"]:
        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"
        os.makedirs(path_plots, exist_ok=True)
        lu.print_blue("Load 26X V19 predictions for 14X J17 and PSNID")
        # True ensemble uncertainties
        pred_ensemble_J17_dic[nn] = get_ensemble_pred(
            f"{args.path_models14X}/models_J17/PRED_{nn}*set*.pickle"
        )
        pred_ensemble_PSNID_dic[nn] = get_ensemble_pred(
            f"{args.path_models14X}/models_PSNID/PRED_{nn}*set*.pickle"
        )

        # Single seed + approx ensemble
        lu.print_blue("Load 14X V19 predictions for J17 and PSNID")
        df_dic_J17[nn] = du.load_merge_all_preds(
            path_class=f"{args.path_models14X}/models_J17/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )
        df_dic_PSNID[nn] = du.load_merge_all_preds(
            path_class=f"{args.path_models14X}/models_PSNID/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )

        # Average uncertainties
        lu.print_blue(nn_decode_dic[nn])
        # Uncertainties V19
        lu.print_green("V19 (26XB)")
        unc_average_behaviour(df_dic[nn]["cosmo_quantile"], pred_ensemble_26XB_dic[nn])
        lu.print_green("PSNID (14X)")
        unc_average_behaviour(
            df_dic_PSNID[nn]["cosmo_quantile"], pred_ensemble_PSNID_dic[nn],
        )
        lu.print_green("J17 (14X)")
        unc_average_behaviour(
            df_dic_J17[nn]["cosmo_quantile"], pred_ensemble_J17_dic[nn]
        )
        # Plots
        # singel model hist
        fig = plt.figure()
        plt.hist(
            df_dic[nn]["cosmo_quantile"]["all_class0_std_S_0"],
            label="V19",
            histtype="step",
        )
        plt.hist(
            df_dic_PSNID[nn]["cosmo_quantile"]["all_class0_std_S_0"],
            label="PSNID",
            histtype="step",
        )
        plt.hist(
            df_dic_J17[nn]["cosmo_quantile"]["all_class0_std_S_0"],
            label="J17",
            histtype="step",
        )
        plt.yscale("log")
        plt.xlabel("all_class0_std_S_0")
        plt.legend()
        plt.savefig(f"{path_plots}/hist_uncertainties_singlemodel.png")

        # ensemble true
        fig = plt.figure()
        plt.hist(
            pred_ensemble_26XB_dic[nn]["all_class0_std_set_0"],
            label="V19 ensemble",
            histtype="step",
        )
        plt.hist(
            pred_ensemble_PSNID_dic[nn]["all_class0_std_set_0"],
            label="PSNID ensemble",
            histtype="step",
        )
        plt.hist(
            pred_ensemble_J17_dic[nn]["all_class0_std_set_0"],
            label="J17 ensemble",
            histtype="step",
        )
        plt.yscale("log")
        plt.xlabel("all_class0_std set 0")
        plt.legend()
        plt.savefig(f"{path_plots}/hist_uncertainties_ensemble.png")

    # For data
    lu.print_blue("Load", "DATA PREDS for J17 and PSNID")
    preds_J17 = {}
    preds_PSNID = {}
    for nn in ["variational", "bayesian"]:
        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"
        os.makedirs(path_plots, exist_ok=True)
        lu.print_blue("Load DATA predictions with J17 and PSNID trained models")
        # Single seed + approx ensemble
        tmp = du.load_merge_all_preds(
            path_class=f"{args.path_data_class}/snndump_14X_J17/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1050*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )
        preds_J17[nn] = tmp["cosmo_quantile"]

        tmp2 = du.load_merge_all_preds(
            path_class=f"{args.path_data_class}/snndump_14X_PSNID/",
            model_name=f"{nn}_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1050*",
            norm="cosmo_quantile",
            prob_key="all_class0_median",
        )
        preds_PSNID[nn] = tmp2["cosmo_quantile"]

        # Average uncertainties
        lu.print_blue(nn_decode_dic[nn])
        du.print_stats(preds[nn]["all_class0_std_S_0"], context="V19")
        du.print_stats(preds_J17[nn]["all_class0_std_S_0"], context="J17")
        du.print_stats(preds_PSNID[nn]["all_class0_std_S_0"], context="PSNID")

        xlabel = r"$\sigma_P"

        fig = plt.figure()
        plt.hist(preds[nn]["all_class0_std_S_0"], histtype="step", label="V19")
        plt.hist(preds_J17[nn]["all_class0_std_S_0"], histtype="step", label="J17")
        plt.hist(preds_PSNID[nn]["all_class0_std_S_0"], histtype="step", label="PSNID")
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.legend()
        plt.savefig(f"{path_plots}/hist_uncertainties_data_all.png")
        del fig

    lu.print_blue("Uncertainties behaviour vs. selection cuts")
    #
    # In depth selection cuts analysis
    #
    # SIM
    for nn in ["variational", "bayesian"]:
        df = df_dic[nn]["cosmo_quantile"]
        unc_fits = df[df.SNID.isin(sim_fits.SNID.values)][
            "average_probability_set_0_meanstd"
        ].values
        unc_JLA = df[df.SNID.isin(sim_JLA_fits.SNID.values)][
            "average_probability_set_0_meanstd"
        ].values
        lu.print_green(f"Simulation uncertainties", nn)
        du.print_stats(
            df["average_probability_set_0_meanstd"].values, context="no cuts"
        )
        du.print_stats(unc_fits, context="with fit")
        du.print_stats(unc_JLA, context="JLA-cuts")

        fig = plt.figure()
        plt.hist(df["average_probability_set_0_meanstd"].values, label="no cuts")
        plt.hist(unc_fits, label="salt fit")
        plt.hist(unc_JLA, label="JLA cuts")
        plt.yscale("log")
        plt.legend()
        path_plots = f"{path_dump}/plots_sample_{nn_decode_dic[nn]}/"
        plt.savefig(f"{path_plots}/hist_uncertainty_sim.png")
        print(f"saved {path_plots}/hist_uncertainty_sim.png")
        del fig

    print("")
    lu.print_green("DATA without cuts")
    df_metadata = du.load_headers(path_data)
    preds_w_metadata = {}
    for nn in ["variational", "bayesian"]:
        preds_w_metadata[nn] = pd.merge(
            data_preds[nn]["cosmo_quantile"], df_metadata, on=["SNID"], how="left"
        )
        preds_w_metadata[nn] = pd.merge(
            preds_w_metadata[nn], data_fits, on=["SNID", "SNTYPE"], how="left"
        )

    path_plots = f"{path_dump}/plots_sample_bothBNNs/"
    os.makedirs(path_plots, exist_ok=True)
    # Uncertainties evolution in the sample
    plot_uncertainties_cuts(
        preds, preds_w_metadata, preds_for_RNN_sample, path_plots, suffix="data"
    )

    print("")
    lu.print_green("APPENDIX")
    # get list of SNIDs to have in table
    # must include RNN,BBB,MC S_0, set_0
    SNIDs_allsamples = np.concatenate(
        [
            sample_baseline.SNID.values,
            sample_baseline_S_0.SNID.values,
            sample_S_0["variational"].SNID.values,
            sample_set_0["variational"].SNID.values,
            sample_S_0["bayesian"].SNID.values,
            sample_set_0["bayesian"].SNID.values,
        ]
    )
    SNIDs_allsamples_list = list(set(SNIDs_allsamples))
    # load predictions RNN
    # load predictions
    cmd = f"{DES}/DES5YR/data_preds/snndump_26XBOOSTEDDES/models/vanilla_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/PRED_*.pickle"
    list_path_p = glob.glob(cmd)
    preds_RNN = {}
    list_df_preds = []
    # SNIDs are the same for all preds
    for path_p in list_path_p:
        seed = re.search(r"(?<=S\_)\d+", path_p).group()
        list_df_preds.append(du.load_preds_addsuffix(path_p, suffix=f"S_{seed}"))
    # merge all predictions
    preds_RNN["cosmo_quantile"] = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=["SNID", "target"],), list_df_preds
    )
    # ensemble methods + metadata
    preds_RNN, list_sets = du.add_ensemble_methods(preds_RNN, "cosmo_quantile")
    print("Predictions for", list_sets)
    df_out = pd.DataFrame()
    df_out["SNID"] = SNIDs_allsamples_list
    # add DES official name
    df_out = pd.merge(df_out, df_metadata[["SNID", "IAUC"]], on="SNID", how="left")
    # add preds from different methods
    def add_preds_out(dfout, dfpreds, prefix=""):
        df_tmp = pd.DataFrame()
        list_of_cols = [
            f"{prefix}_{k}"
            for k in [
                "pIa_0",
                "pIa_55",
                "pIa_100",
                "pIa_1000",
                "pIa_30469",
                "pIa_set0",
            ]
        ]
        df_tmp[["SNID"] + list_of_cols] = dfpreds[
            [
                "SNID",
                "all_class0_S_0",
                "all_class0_S_55",
                "all_class0_S_100",
                "all_class0_S_1000",
                "all_class0_S_30469",
                "average_probability_set_0",
            ]
        ]
        dfout = pd.merge(dfout, df_tmp, on="SNID", how="left")
        return dfout

    df_out = add_preds_out(df_out, preds_RNN["cosmo_quantile"], prefix="RNN")
    df_out = add_preds_out(df_out, preds["variational"], prefix="MC")
    df_out = add_preds_out(df_out, preds["bayesian"], prefix="BBB")
    df_out = df_out.round(2)

    cols_to_print = [k for k in df_out.keys() if k != "SNID"]
    latex_table = df_out[cols_to_print].to_latex(
        buf=f"{path_dump}/DES5year_PIa_table.tex", index=False
    )
    df_out[cols_to_print].to_csv(f"{path_dump}/DES5year_PIa_table.csv", index=False)

