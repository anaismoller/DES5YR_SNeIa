import logging
import argparse
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import metric_utils as mu
from utils import science_utils as su
from utils import logging_utils as lu
from utils import utils_emcee_poisson as mc

plt.switch_backend("agg")

"""
TO REPRODUCE
- data: pippin/AM_DATA5YR.yml
- models: trained 1g. 26XB
- predictions: reproduce/1h.SNN_26XDESBOSSTED_DATA.sh


"""


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
        "--path_data_class",
        type=str,
        default=f"{DES}/DES5YR/data_preds/snndump_26XBOOSTEDDES/models/",
        help="Path to data predictions",
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
    parser.add_argument(
        "--path_dump",
        default=f"./dump_DES5YR",
        type=str,
        help="Path to output & sample",
    )
    parser.add_argument(
        "--nofit", action="store_true", help="if no fit to selection function",
    )

    # Init
    args = parser.parse_args()
    path_data_fits = args.path_data_fits
    path_data_class = args.path_data_class
    path_sim_fits = args.path_sim_fits
    path_dump = args.path_dump
    path_sim_class = args.path_sim_class

    path_plots = f"{path_dump}/plots_sample_wz/"
    os.makedirs(path_plots, exist_ok=True)
    path_samples = f"{path_dump}/samples"

    # logger
    logger = setup_logging()

    # save configuration
    logger.info("_______________")
    logger.info("CONFIGURATION")
    logger.info(f"{args}")  # --path_sims {path_sims}"

    # Load sample
    path_sample = f"{path_samples}/photoIa_cosmo_quantile_average_probability_set_0.csv"
    lu.print_blue("Loading sample & fits", path_sample)
    sample = pd.read_csv(path_sample)
    data_fits = du.load_salt_fits(path_data_fits)
    # select only events that are in the selected photometric sample
    sample_fits = data_fits[data_fits.SNID.isin(sample.SNID)]
    lu.print_green(f"Photo Ia sample {len(sample)} fits: {len(sample_fits)}")

    lu.print_blue("Loading data preds cosmo_quantile", path_sim_fits)
    data_preds = du.load_merge_all_preds(
        path_class=path_data_class,
        model_name="vanilla_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )

    # Load 5X simulation fits
    lu.print_blue("Loading sim fits", path_sim_fits)
    sim_fits = du.load_salt_fits(path_sim_fits)

    # same redshift cut than data
    sim_fits = sim_fits[(sim_fits["zHD"] > 0.05) & (sim_fits["zHD"] < 1.3)]
    # now options
    sim_JLA_fits = su.apply_JLA_cut(sim_fits)
    # select true SNe Ia and apply same JLA cuts than sample
    sim_Ia_fits = sim_fits[sim_fits.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_Ia_fits_JLA = su.apply_JLA_cut(sim_Ia_fits)

    lu.print_blue("Loading sim preds cosmo_quantile", path_sim_class)
    sim_preds = du.load_merge_all_preds(path_class=path_sim_class)

    # evaluate performance of our classification model in independent sims
    sim_wfits = pd.merge(sim_fits, sim_preds["cosmo_quantile"])
    sim_wfits_JLA = pd.merge(sim_JLA_fits, sim_preds["cosmo_quantile"])
    df_txt_stats_sel = pd.DataFrame(
        columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
    )
    df_txt_stats_JLA = pd.DataFrame(
        columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
    )
    for method, desc in cu.dic_sel_methods.items():
        list_seeds_sets = (
            cu.list_seeds_set[0] if method == "predicted_target_S_" else cu.list_sets
        )
        # sel cuts only
        df_txt_stats_sel = mu.get_multiseed_performance_metrics(
            sim_wfits,
            key_pred_targ_prefix=method,
            list_seeds=list_seeds_sets,
            df_txt=df_txt_stats_sel,
            dic_prefilled_keywords={
                "norm": "cosmo_quantile",
                "dataset": "selection cuts",
                "method": desc,
            },
        )
        # JLA only
        df_txt_stats_JLA = mu.get_multiseed_performance_metrics(
            sim_wfits_JLA,
            key_pred_targ_prefix=method,
            list_seeds=list_seeds_sets,
            df_txt=df_txt_stats_JLA,
            dic_prefilled_keywords={
                "norm": "cosmo_quantile",
                "dataset": "JLA",
                "method": desc,
            },
        )
    lu.print_green("Performance on 5XDES with cuts")
    print("Selection cuts")
    print(
        df_txt_stats_sel[["method", "accuracy", "efficiency", "purity"]].to_latex(
            index=False
        )
    )
    print("JLA cuts")
    print(
        df_txt_stats_JLA[["method", "accuracy", "efficiency", "purity"]].to_latex(
            index=False
        )
    )

    sim_cut_photoIa = sim_preds["cosmo_quantile"]["average_probability_set_0"] > 0.5
    sim_photoIa_fits = sim_JLA_fits[
        sim_JLA_fits.SNID.isin(sim_preds["cosmo_quantile"][sim_cut_photoIa].SNID.values)
    ]
    lu.print_yellow(
        "Assuming that redshift quality + multiseason cuts are not necessary in sims"
    )

    logger.info("")
    logger.info("Check selection function")
    lu.print_yellow("Using simulated SNe Ia")
    print("- with converging SALT2 fit")
    print("- using HOST efficiency from M. Vincenzi")

    # Selection function
    variable = "m0obs_i"
    quant = 0.01
    min_var = sample_fits[variable].quantile(quant)
    lu.print_yellow(
        f"Not using photo Ias with {variable}<{min_var} (equivalent to quantile {quant}, possible if low-z)"
    )
    df, minv, maxv = du.data_sim_ratio(
        sample_fits,
        sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        norm=1 / 150,  # the scale of the simulations vs. DES
    )
    if args.nofit:
        lu.print_red("Not doing sel function emcee_fitting!!!!")
    else:
        # Emcee fit of dat/sim
        theta_mcmc, min_theta_mcmc, max_theta_mcmc = mc.emcee_fitting(
            df, path_plots, min_var=min_var
        )

    # sanity ratio z
    variable = "zHD"
    quant = 0.01
    min_var = sample_fits[variable].quantile(quant)
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        sample_fits,
        sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
    )

    logger.info("")
    logger.info("General properties of sample & sim")
    # histo c,x1,z and c,x1 evolution by redshift
    pu.overplot_salt_distributions_lists(
        [sim_Ia_fits_JLA, sample_fits],
        list_labels=["sim Ia JLA", "Baseline DES JLA",],
        path_plots=path_plots,
    )
    # add info wether is deep fields or not
    sim_Ia_fits_JLA["deep"] = sim_Ia_fits_JLA["FIELD"].apply(
        lambda row: any(f in row for f in ["X3", "C3"])
    )
    sample_fits["deep"] = sample_fits["FIELD"].apply(
        lambda row: any(f in row for f in ["X3", "C3"])
    )
    # deep
    sim_Ia_fits_JLA_deep = sim_Ia_fits_JLA[sim_Ia_fits_JLA["deep"] == True]
    sample_fits_deep = sample_fits[sample_fits["deep"] == True]
    # shallow fields
    sim_Ia_fits_JLA_shallow = sim_Ia_fits_JLA[sim_Ia_fits_JLA["deep"] != True]
    sample_fits_shallow = sample_fits[sample_fits["deep"] != True]
    pu.overplot_salt_distributions_lists_deep_shallow(
        [sim_Ia_fits_JLA, sample_fits,],
        list_labels=["sim Ia JLA", "Baseline DES JLA",],
        path_plots=path_plots,
        suffix="deep_and_shallow_fields",
    )
    print("n events in each zbin")
    sample_fits.groupby("zHD_bin").count()["SNID"]

    logger.info("")
    logger.info("Effect of selection cuts")
    lu.print_blue("Loading data with only selection cuts")
    path_sample_selcuts = f"{path_samples}/only_selection_cuts.csv"
    tmp_sample_selcuts_only = pd.read_csv(path_sample_selcuts)
    selcuts_fits = data_fits[data_fits.SNID.isin(tmp_sample_selcuts_only.SNID.values)]

    logger.info("")
    logger.info("Effect of SNN models")
    path_plots_allmodels = f"{path_dump}/plots_sample/all_models/"
    os.makedirs(path_plots_allmodels, exist_ok=True)
    # on simulated SNe
    # contamination against all sims that go into photoIa
    pu.plot_contamination_list([(sim_preds, sim_JLA_fits)], path_plots=path_plots)
    # sim: check if there variations with different classification models
    pu.overplot_salt_distributions_allmodels(
        sim_fits,
        sim_preds["cosmo_quantile"],
        path_plots=path_plots_allmodels,
        namestr="sim",
    )

    # on data
    # after selection cuts!!!!!
    pu.overplot_salt_distributions_allmodels(
        selcuts_fits,
        data_preds["cosmo_quantile"],
        path_plots=path_plots_allmodels,
        namestr="selcuts",
    )

    # HD
    merged_sample_fits_with_preds = pd.merge(
        sample_fits, data_preds["cosmo_quantile"], on="SNID"
    )

    pu.plot_HD_residuals(
        merged_sample_fits_with_preds,
        f"{path_plots}/HDres.png",
        prob_key="average_probability_set_0",
    )
