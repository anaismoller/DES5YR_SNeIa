import re
import os, glob
import json
import ipdb
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from . import data_utils as du
from . import conf_utils as cu
from . import plot_utils as pu
from astropy.table import Table
from . import logging_utils as lu
from utils import science_utils as su


"""All selection cuts are defined here
"""

inv_spec_tags = {}
for k, v in cu.spec_tags.items():
    for tmp in v:
        inv_spec_tags[tmp] = k

#
# USEFUL FUNCTIONS
#
def message_cut(logger, message):
    logger.info("")
    logger.info(f"{message}")


def make_reduction(df, cut, logger):
    """Function to cut and log reduction

    Args:
        df (pd.DataFrame): DataFrame to perform cuts
        cut (sel pd.DataFrame): cut to perform

    Returns:
        df_with_new_cut (pd.DataFrame): DataFrame with cuts
    """
    df_with_new_cut = df[cut]

    logger.info(f"#lcs {len(df)} reduced to {len(df_with_new_cut)}")

    return df_with_new_cut


def do_arr_stats(arr):
    """Mean +- std of an array into dictionary
    """
    arr_mean = np.array(arr).mean(axis=0).astype(int)
    arr_min = np.array(arr).min(axis=0).astype(int)
    arr_max = np.array(arr).max(axis=0).astype(int)
    photoIa = f"{arr_mean[0]}+{arr_max[0] - arr_mean[0]}-{arr_mean[0]-arr_min[0]}"
    specIa = f"{arr_mean[1]}+{arr_max[1] - arr_mean[1]}-{arr_mean[1]-arr_min[1]}"
    specCC = f"{arr_mean[2]}+{arr_max[2] - arr_mean[2]}-{arr_mean[2]-arr_min[2]}"
    specother = f"{arr_mean[3]}+{arr_max[3] - arr_mean[3]}-{arr_mean[3]-arr_min[3]}"

    return photoIa, specIa, specCC, specother


def photo_sel_prob(
    df, prob_threshold=0.5, df_out=False, prob_key="all_class0", do_prob_thres=True
):
    """Generic photoIa and spec subsamples with probability cut

    Args:
        df (pd.DataFrame): DataFrame to perform cuts
        prob_threshold (float): threshold for sample selection
        df_out (boolean): if True return the DataFrame, else just lengths
        prob_key (str): probability to apply threshold on for selection
        do_prob_thres (Boolean): apply a probability threshold
    Returns:

    """

    # spec_samples
    cut_spec_Ia = df["SNTYPE"].isin(cu.spec_tags["Ia"])
    cut_spec_CC = df["SNTYPE"].isin(cu.spec_tags["CC"])
    cut_spec_other = ~df["SNTYPE"].isin(
        cu.spec_tags["Ia"] + cu.spec_tags["CC"] + [0]
    )  # 0 is unknown

    # photo sample
    if do_prob_thres:
        cut_prob = df[prob_key] > prob_threshold
        photoIa = df[cut_prob]
        specIa = df[cut_prob & cut_spec_Ia]
        specCC = df[cut_prob & cut_spec_CC]
        specother = df[cut_prob & cut_spec_other]
    else:
        photoIa = df
        specIa = df[cut_spec_Ia]
        specCC = df[cut_spec_CC]
        specother = df[cut_spec_other]

    if df_out:
        return photoIa, specIa, specCC, specother
    else:
        return len(photoIa), len(specIa), len(specCC), len(specother)


def photo_sel_target(df, target_goal=0, df_out=False, target_key="predicted_target"):
    """Generic photoIa and spec subsamples with target cut

    Args:
        df (pd.DataFrame): DataFrame to perform cuts
        target_goal (int): which target is selected
        df_out (boolean): if True return the DataFrame, else just lengths
        target_key (str): target key for selection
    Returns:

    """

    # spec_samples
    cut_spec_Ia = df["SNTYPE"].isin(cu.spec_tags["Ia"])
    cut_spec_CC = df["SNTYPE"].isin(cu.spec_tags["CC"])
    cut_spec_other = ~df["SNTYPE"].isin(
        cu.spec_tags["Ia"] + cu.spec_tags["CC"] + [0]
    )  # 0 is unknown

    # photo sample
    cut_prob = df[target_key] == target_goal

    photoIa = df[cut_prob]
    specIa = df[cut_prob & cut_spec_Ia]
    specCC = df[cut_prob & cut_spec_CC]
    specother = df[cut_prob & cut_spec_other]

    if df_out:
        return photoIa, specIa, specCC, specother
    else:
        return len(photoIa), len(specIa), len(specCC), len(specother)


def spec_subsamples(df_metadata, logger):
    """Log spec subsamples
    
    Args:
        df (pd.DataFrame): DataFrame to perform cuts
        logger (obj): logger
    """

    sntypes = df_metadata.groupby("SNTYPE").count()["SNID"]

    list_spec_types = []
    for st_type, list_sntypes in cu.spec_tags.items():
        n = sntypes[sntypes.index.isin(list_sntypes)].sum()
        if st_type == "Ia":
            specIa = n
        else:
            list_spec_types.append(f"{st_type}:{n}")

    DES3yr = len(
        df_metadata[
            (df_metadata["IAUC"].str.contains("DES15|DES14|DES13"))
            & (df_metadata["SNTYPE"].isin(cu.spec_tags["Ia"]))
        ]
    )

    logger.info(f"#spec Ia: {specIa} (DES3yr:{DES3yr}), {list_spec_types}")


#
# CUTS START HERE
#
def data_processing(message, path_data, logger):
    """Data processing cut: photometry quality
    Criteria: non-reliable photometry using bit-map

    Args:
        message (str): name of cut
        file_path (str): path to data ``.FITS`` files
        logger (obj): logger
    """
    message_cut(logger, message)

    # Load photometry
    df_data_phot = du.load_photometry(path_data)

    # Quality is a bit map of powers of two
    def powers_of_two(x):
        powers = []
        i = 1
        while i <= x:
            if i & x:
                powers.append(i)
            i <<= 1
        return powers

    # flag rejected photometry
    df_data_phot["phot_reject"] = df_data_phot["PHOTFLAG"].apply(
        lambda x: False
        if len(set([8, 16, 32, 64, 128, 256, 512]).intersection(set(powers_of_two(x))))
        > 0
        else True
    )
    # remove unreliable photometry
    tmp = df_data_phot[df_data_phot["phot_reject"] == True]

    # Save stats
    logger.info("")
    logger.info(">> Photometry quality (PHOTFLAG) cut")
    logger.info(
        f"# light-curves {len(df_data_phot.SNID.unique())} reduced to {len(tmp.SNID.unique())}"
    )
    logger.info(
        f"# photometric points {len(df_data_phot)} reduced to {len(tmp)} (by {round(100-(100*len(tmp)/len(df_data_phot)),2)}%)"
    )
    del df_data_phot, tmp


def redshift(message, df_metadata, logger):
    """Redshift cuts
    REDSHIFT_FINAL is the best available redshift
    it is either from the SN spectra or HOSTGAL_SPECZ
    all HOSTGAL_SPECZ are high-quality or are available from PRIMUS (lower quality)
    all are DLR <7 (as Vincenzi 2020)

    Criteria: redshift quality and range

    Args:
        message (str): name of cut
        df_metadata (pd.DataFrame): data frame to cut
        logger (obj): logger
    """
    message_cut(logger, message)

    logger.info(">> Assigned quality redshift")
    assigned_redshifts = df_metadata.REDSHIFT_FINAL != -9.0
    df_metadata_sel = make_reduction(df_metadata, assigned_redshifts, logger)

    logger.info(">> Redshift range (eliminates stars and AGNs)")
    redshift_range = df_metadata_sel.REDSHIFT_FINAL > 0.05
    df_metadata_sel = make_reduction(df_metadata_sel, redshift_range, logger)

    spec_subsamples(df_metadata_sel, logger)

    return df_metadata_sel


def detections(message, df_metadata, logger):
    """Cut on detection for spurious + multi-season
    DES_numepochs_ml = Number of distinct nights with a detection that pass ML
    Given that the cadence is ~7 days - 4 epochs is ~a month which sounds right for something at say z=0.7

    Args:
        message (str): name of cut
        df_metadata (pd.DataFrame): data frame to cut
        logger (obj): logger
    """
    message_cut(logger, message)

    # logger.info(">> Detections (3<=det)")
    det_ml_cut = df_metadata["PRIVATE(DES_numepochs_ml)"] >= 3.0
    df_metadata_sel = make_reduction(df_metadata, det_ml_cut, logger)
    # spec_subsamples(df_metadata_sel, logger)

    logger.info(
        ">> Detections + multi-season (3<=det & num_epochs_ml/num_epochs > 0.2)"
    )
    df_metadata["ratio_numepochs"] = (
        df_metadata["PRIVATE(DES_numepochs_ml)"] / df_metadata["PRIVATE(DES_numepochs)"]
    )
    det_ml_cut = (df_metadata["PRIVATE(DES_numepochs_ml)"] >= 3.0) & (
        df_metadata["ratio_numepochs"] > 0.2
    )
    df_metadata_sel = make_reduction(df_metadata, det_ml_cut, logger)
    spec_subsamples(df_metadata_sel, logger)

    return df_metadata_sel


def transient_status(message, df_metadata, logger):
    """Multi-season cuts (non-exhaustive)
    TRANSIENT_STATUS: single season transient from detections (used in real-time processing)

    Args:
        message (str): name of cut
        df_metadata (pd.DataFrame): data frame to cut
        logger (obj): logger
    """
    message_cut(logger, message)
    single_seasons = [1.0, 2.0, 4.0, 8.0, 16.0]

    logger.info(">> Transient status")
    assigned_redshifts = df_metadata["PRIVATE(DES_transient_status)"].isin(
        single_seasons
    )
    df_metadata_sel = make_reduction(df_metadata, assigned_redshifts, logger)

    spec_subsamples(df_metadata_sel, logger)

    return df_metadata_sel


def salt_basic(message, df_metadata, df_salt, logger):
    """SALT sampling + convergence
    SALT2 fits require a minimal sampling
    see: https://github.com/Samreay/Pippin/blob/master/data_files/surveys/des/lcfit_nml/des_5yr.nml 
    cuts on sampling and redshift range

    Args:
        message (str): name of cut
        df_metadata (pd.DataFrame): data frame to cut
        file_path (str): path to SALT fits
        logger (obj): logger
    """
    message_cut(logger, message)

    SNID_salt_converges = df_salt.SNID.tolist()

    salt_converges = df_metadata["SNID"].isin(SNID_salt_converges)
    df_metadata_sel = make_reduction(df_metadata, salt_converges, logger)

    # basic stats
    spec_subsamples(df_metadata_sel, logger)

    # keep SALT fit values
    df_salt_sel = df_salt[
        [
            "SNID",
            "SNTYPE",
            "FIELD",
            "CUTFLAG_SNANA",
            "zHEL",
            "zHELERR",
            "zCMB",
            "zCMBERR",
            "zHD",
            "zHDERR",
            "VPEC",
            "VPECERR",
            "MWEBV",
            "HOST_LOGMASS",
            "HOST_LOGMASS_ERR",
            "HOST_sSFR",
            "HOST_sSFR_ERR",
            "PKMJDINI",
            "SNRMAX1",
            "SNRMAX2",
            "SNRMAX3",
            "PKMJD",
            "PKMJDERR",
            "x1",
            "x1ERR",
            "c",
            "cERR",
            "mB",
            "mBERR",
            "x0",
            "x0ERR",
            "COV_x1_c",
            "COV_x1_x0",
            "COV_c_x0",
            "NDOF",
            "FITCHI2",
            "FITPROB",
        ]
    ]
    df_metadata_sel = pd.merge(
        df_metadata_sel, df_salt_sel, on=["SNID", "SNTYPE"], how="left"
    )

    return df_metadata_sel


def photo_norm(df_metadata, path_class, path_dump, logger, z="zspe", path_plots="."):
    """Photometric sample varying norm in SNN (host-zspe)

    Args:
        df_metadata (pd.DataFrame): data frame to cut
        path_class (str): path to predictions and trained models
        path_dump (str): path to dump folder
        logger (obj): logger
        z (str): zspe= spectroscopic host-gal redshift, none=no z

    Returns:
        photo_Ia_wsalt (dict of pd.DataFrame): photometrically selected samples w salt fit
        df_photoIa_stats (pd.DataFrame): stats of selection
    """

    if z == "zspe":
        prefix = "wz"
    elif z == "zpho":
        prefix = "wzpho"
    elif z == "none":
        prefix = "nz"

    # select sample
    norm_list = ["cosmo", "cosmo_quantile"]
    df_dic = {}
    for norm in norm_list:
        # load predictions
        cmd = f"{path_class}/vanilla_S_*_{z}*_{norm}_lstm_64x4_0.05_1024_True_mean/PRED_*.pickle"
        list_path_p = glob.glob(cmd)
        list_df_preds = []
        # SNIDs are the same for all preds
        for path_p in list_path_p:
            seed = re.search(r"(?<=S\_)\d+", path_p).group()
            list_df_preds.append(du.load_preds_addsuffix(path_p, suffix=f"S_{seed}"))

        # merge all predictions
        df_dic[norm] = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=["SNID", "target"],), list_df_preds
        )
        # ensemble methods + metadata
        df_dic, list_sets = du.add_ensemble_methods(df_dic, norm)
        print("Predictions for", list_sets)

    # Need to compute this after both norms have been filled
    df_photoIa_stats = pd.DataFrame(
        columns=["norm", "method", "photoIa", "specIa", "specCC", "specOther"]
    )
    df_dic_wsalt = {}
    df_photoIa_wsalt = {"cosmo": {}, "cosmo_quantile": {}}
    for norm in norm_list:
        # only lcs that have salt convergent fits
        df_dic_wsalt[norm] = pd.merge(df_metadata, df_dic[norm], how="left", on="SNID")

        # stats
        for method in [
            "single_model",
            "average_probability",
        ]:
            arr = []
            if method == "single_model":
                for seed in cu.list_seeds_set[0]:
                    # save photo Ia
                    df_photoIa_wsalt[norm][f"S_{seed}"], *_ = photo_sel_prob(
                        df_dic_wsalt[norm], df_out=True, prob_key=f"all_class0_S_{seed}"
                    )
                    # get size samples
                    arr.append(
                        photo_sel_prob(
                            df_dic_wsalt[norm], prob_key=f"all_class0_S_{seed}"
                        )
                    )
            else:
                for modelset in list_sets:
                    # save photo Ia
                    (
                        df_photoIa_wsalt[norm][f"{method}_set_{modelset}"],
                        *_,
                    ) = photo_sel_target(
                        df_dic_wsalt[norm],
                        target_key=f"predicted_target_{method}_set_{modelset}",
                        df_out=True,
                    )
                    arr.append(
                        photo_sel_target(
                            df_dic_wsalt[norm],
                            target_key=f"predicted_target_{method}_set_{modelset}",
                        )
                    )

            photoIa, specIa, specCC, specother = do_arr_stats(arr)
            dic_tmp = {
                "norm": norm,
                "method": method.replace("_", " "),
                "photoIa": photoIa,
                "specIa": specIa,
                "specCC": specCC,
                "specOther": specother,
            }
            df_photoIa_stats = df_photoIa_stats.append(dic_tmp, ignore_index=True)
        # set 0 average probability sample
        photoIa_avg_prob_set_0 = photo_sel_target(
            df_dic_wsalt[norm],
            target_key=f"predicted_target_average_probability_set_0",
        )
        dic_tmp = {
            "norm": norm,
            "method": "photo Ia no z av.prob. set 0",
            "photoIa": photoIa_avg_prob_set_0[0],
            "specIa": photoIa_avg_prob_set_0[1],
            "specCC": photoIa_avg_prob_set_0[2],
            "specOther": photoIa_avg_prob_set_0[3],
        }
        df_photoIa_stats = df_photoIa_stats.append(dic_tmp, ignore_index=True)

    # Latex table
    for norm in norm_list:
        lu.print_blue(norm)
        latex_table = df_photoIa_stats[df_photoIa_stats["norm"] == norm].to_latex(
            index=False, columns=["method", "photoIa", "specIa", "specCC", "specOther"]
        )
        logger.info(latex_table)

    # Venn diagrams
    dic_venn = {}
    for norm in ["cosmo", "cosmo_quantile"]:
        # all
        dic_venn[f"{prefix}_{norm}_ens_all"] = set(df_dic_wsalt[norm].SNID.unique())
        # seeds for set0
        for seed in cu.list_seeds_set[0]:
            if norm in df_dic_wsalt.keys():
                dic_venn[f"{prefix}_{norm}_seed_{seed}"] = set(
                    df_dic_wsalt[norm][
                        df_dic_wsalt[norm][f"predicted_target_S_{seed}"] == 0
                    ].SNID.unique()
                )
        # ensemble
        for set_model_average in [0]:
            for ensmethod in ["average_target", "average_probability"]:
                dic_venn[f"{prefix}_{norm}_ens_{ensmethod}_{set_model_average}"] = set(
                    df_dic_wsalt[norm][
                        df_dic_wsalt[norm][
                            f"predicted_target_{ensmethod}_set_{set_model_average}"
                        ]
                        == 0
                    ].SNID.unique()
                )

    for norm in ["cosmo", "cosmo_quantile"]:
        # seeds (only set0)
        pu.plot_venn_percentages(
            dict(
                (k, dic_venn[k])
                for k in dic_venn.keys()
                if f"{prefix}_{norm}_seed" in k
            ),
            path_plots=f"{path_plots}/",
            suffix=f"data_seeds_{norm}_{prefix}",
        )
        # ensemble
        pu.plot_venn_percentages(
            dict(
                (k, dic_venn[k]) for k in dic_venn.keys() if f"{prefix}_{norm}_ens" in k
            ),
            path_plots=f"{path_plots}/",
            suffix=f"data_ensembles_{norm}_{prefix}",
        )

    return df_photoIa_wsalt, df_photoIa_stats


def apply_salt_cuts(df, SNID_to_keep, logger, verbose=False):
    """ Only salt cuts
    Args:
        df (pd.DataFrame): data to apply cuts
        SNID_to_keep (list): SNIDs that have high-quality z
    Return:
        df_JLA_zHQ (pd.DataFrame): data with all cuts

    """

    # SALT cuts
    cut_salt_loose = (
        (df.x1 > -4.9)
        & (df.x1 < 4.9)
        & (df.c > -0.49)
        & (df.c < 0.49)
        & (df.FITPROB > 0.001)
    )

    df_loose = df[cut_salt_loose]

    df_JLA = su.apply_JLA_cut(df)

    df_JLA_zHQ = df_JLA[df_JLA.SNID.isin(SNID_to_keep)]

    if verbose:

        logger.info("loose SALT")
        logger.info(f"#lcs {len(df)} reduced to {len(df_loose)}")
        spec_subsamples(df_loose, logger)

        logger.info("JLA SALT")
        logger.info(f"#lcs {len(df_loose)} reduced to {len(df_JLA)}")
        spec_subsamples(df_JLA, logger)

        logger.info("JLA SALT + z HQ")
        logger.info(f"#lcs {len(df_JLA)} reduced to {len(df_JLA_zHQ)}")
        spec_subsamples(df_JLA_zHQ, logger)

        logger.info("only PRIMUS eliminated")
        logger.info(f"#lcs {len(df)} reduced to {len(df[df.SNID.isin(SNID_to_keep)])}")
        spec_subsamples(df[df.SNID.isin(SNID_to_keep)], logger)

    return df_JLA_zHQ


def towards_cosmo(dic_df_photoIa_wsalt, df_photoIa_stats, logger):
    """ SALT2 cuts + redshift extra quality cuts

    Args:
        dic_df_photoIa_wsalt (pd.DataFrame): photometrically classified sample w salt fits
        df_photoIa_stats (pd.DataFrame): pre-filled table with stats
        logger (obj): logger

    Returns:
    """

    # Load redshift info
    hostz_info = pd.read_csv("extra_lists/SNGALS_DLR_RANK1_INFO.csv")
    hostz_info["SPECZ_CATALOG"] = hostz_info.SPECZ_CATALOG.apply(
        lambda x: x[2:-1].strip(" ")
    )
    SNID_to_keep = hostz_info[hostz_info.SPECZ_CATALOG != "PRIMUS"].SNID.values.tolist()

    norm_list = ["cosmo", "cosmo_quantile"]

    dic_photoIa_sel = {}

    # lu.print_blue("Stats from photometric samples")
    for norm in norm_list:
        dic_photoIa_sel[norm] = {}
        for k in dic_df_photoIa_wsalt[norm].keys():
            df = dic_df_photoIa_wsalt[norm][k]
            # fetch JLA+ zHQ
            dic_photoIa_sel[norm][k] = apply_salt_cuts(
                df, SNID_to_keep, logger, verbose=False
            )

    # Compute samples with margins
    lu.print_blue("photo + SALT + zHQ cuts")
    df_photoIa_stats_tmp = pd.DataFrame(
        columns=[
            "norm",
            "method",
            "photoIa_JLA_zHQ",
            "specIa_JLA_zHQ",
            "specCC_JLA_zHQ",
            "specOther_JLA_zHQ",
        ]
    )
    for norm in norm_list:
        for method in [
            "single_model",
            # "average_target",
            "average_probability",
        ]:
            arr = []
            if method == "single_model":
                for seed in cu.list_seeds_set[0]:
                    df = dic_photoIa_sel[norm][f"S_{seed}"]
                    arr.append(photo_sel_prob(df, do_prob_thres=False))
            else:
                for modelset in cu.list_sets:
                    df = dic_photoIa_sel[norm][f"{method}_set_{modelset}"]
                    arr.append(photo_sel_prob(df, do_prob_thres=False,))
            photoIa, specIa, specCC, specother = do_arr_stats(arr)
            dic_tmp = {
                "norm": norm,
                "method": method.replace("_", " "),
                "photoIa_JLA_zHQ": photoIa,
                "specIa_JLA_zHQ": specIa,
                "specCC_JLA_zHQ": specCC,
                "specOther_JLA_zHQ": specother,
            }
            df_photoIa_stats_tmp = df_photoIa_stats_tmp.append(
                dic_tmp, ignore_index=True
            )
        # set 0 average probability sample
        photoIa_avg_prob_set_0 = photo_sel_prob(
            dic_photoIa_sel[norm]["average_probability_set_0"], do_prob_thres=False,
        )
        dic_tmp = {
            "norm": norm,
            "method": "photo Ia no z av.prob. set 0",
            "photoIa_JLA_zHQ": photoIa_avg_prob_set_0[0],
            "specIa_JLA_zHQ": photoIa_avg_prob_set_0[1],
            "specCC_JLA_zHQ": photoIa_avg_prob_set_0[2],
            "specOther_JLA_zHQ": photoIa_avg_prob_set_0[3],
        }
        df_photoIa_stats_tmp = df_photoIa_stats_tmp.append(dic_tmp, ignore_index=True)

    # Latex table
    df_photoIa_stats = pd.merge(
        df_photoIa_stats, df_photoIa_stats_tmp, on=["norm", "method"], how="left"
    )
    cols_to_print = ["method", "photoIa", "specIa", "photoIa_JLA_zHQ", "specIa_JLA_zHQ"]
    # for norm in norm_list:
    for norm in ["cosmo_quantile"]:
        print("\multicolumn", {len(cols_to_print)}, "{c}{", norm, "}\\\\")
        print("\hline")
        latex_table = df_photoIa_stats[df_photoIa_stats.norm == norm].to_latex(
            index=False, columns=cols_to_print
        )
        logger.info(latex_table)

    return dic_photoIa_sel


def apply_visual_cuts(df, SNID=False):
    cut_ok = (
        (df.comment.str.contains("beaut"))
        | (df.comment.str.contains("nice"))
        | (df.comment.str.contains("ok"))
        | df.comment_noisy.str.contains("ok")
    )

    cut_partial = (df.comment.str.contains("partial")) | df.comment_noisy.str.contains(
        "partial"
    )

    cut_noisy = (
        (~df.comment_noisy.isna()) & (df.comment_noisy.str.contains("noisy"))
    ) | ((df.comment_noisy.isna()) & (df.comment.str.contains("noisy")))

    cut_noise = (
        (~df.comment_noisy.isna()) & (df.comment_noisy.str.contains("noise"))
    ) | (df.comment == "noise")

    cut_sparse = (~df.comment_noisy.isna()) & (
        df.comment_noisy.str.contains("sparse")
    ) | (df.comment == "sparse")

    if SNID:
        return (
            df[cut_ok].SNID.unique().tolist(),
            df[cut_partial].SNID.unique().tolist(),
            df[cut_noisy].SNID.unique().tolist(),
            df[cut_noise].SNID.unique().tolist(),
            df[cut_sparse].SNID.unique().tolist(),
        )
    else:
        return df[cut_ok], df[cut_partial], df[cut_noisy], df[cut_noise], df[cut_sparse]


def cut_visual_inspection_stats(
    df_sel,
    df_visual_inspection,
    df_metadata,
    path_plots="./",
    description_sample="test",
):
    """ Visual inspection cross-match

        Args:
            df_sel (dic of pd.DataFrame): selected metadata
            df_visual_inspection (pd.DataFrame): information from visual inspection
            df_metadata (pd.DataFrame): complete metadata for 5-year
            path_plots (Path): path to save plots
            description_sample (str): sample description

        Returns:


        """

    # Formatting and tag selection
    df_visual_inspection["SNID"] = df_visual_inspection["SNID"].astype(np.int32)
    dic_SNIDs_vi = {}
    (
        dic_SNIDs_vi["ok"],
        dic_SNIDs_vi["partial"],
        dic_SNIDs_vi["noisy"],
        dic_SNIDs_vi["noise"],
        dic_SNIDs_vi["sparse"],
    ) = apply_visual_cuts(df_visual_inspection, SNID=True)

    pu.plot_labels(
        df_metadata, dic_SNIDs_vi, path_plots=path_plots, suffix="summary_inspection",
    )

    df = pd.merge(df_sel, df_visual_inspection, on="SNID", how="left")
    dic_df = {}
    (
        dic_df["ok"],
        dic_df["partial"],
        dic_df["noisy"],
        dic_df["noise"],
        dic_df["sparse"],
    ) = apply_visual_cuts(df)

    print(
        f"{description_sample} total {len(df_sel)}",
        [f"{k}:{len(dic_df[k])}" for k in dic_df.keys()],
    )

    return dic_df


def stats_possible_contaminants(
    dic_photoIa_sel,
    method_list=["S", "average_target_set", "average_probability_set"],
    verbose=True,
):
    """ Check possible other contaminants

    Samples may contain:
    - close-by AGNs (which would deem them not cosmology suited)
    - AGNs
    - other types of SNe

    Statistics of possible detection methods:
    - Spectra: reliable classification but can be just a closeby AGN
    - Visual inspection: useful to select multi-season transients but with human bias
    - Automatic tagging: unreliable due to training sets and possible OOD behaviours

    Available tags:

    # Visual inspections to tag AGNs
    #
    # SCAN_MS1 = Masao's scan of random SNIDs (0: non-SN, 1: SN, -1: unscanned)
    # SCAN_MS2 = Masao's scan of SNRMAX2>=20 SNIDs (0: non-SN, 1: SN, -1: unscanned)
    # SCAN_MV = Maria's scan (0: AGN, 1: SN, 2: DK, -1: unscanned)
    #
    # Automated algorithms (not necessarily reliable)
    # flagSNR3 = 1: single season, 2: adjacent season, 0: otherwise with S/N>3 points
    # flagSNR5 = 1: single season, 2: adjacent season, 0: otherwise with S/N>5 points
    # PROB_RF = random forest probability of good candidate
    # PROB_BDT = boosted decision tree probability
    # COADD_OBJECTS_ID = Phil's matched Quasar ID
    # QSO_SPECZ = Phil's matched Quasar redshift

    """

    # Load OzDES close-by AGN list
    # Transient spectra was inspected for AGN features (see C. Lidman)
    OzDES_AGN_tmp = Table.read("./extra_lists/OzDES_AGN_2020_08_15.fits", format="fits")
    OzDES_AGN = OzDES_AGN_tmp.to_pandas()
    OzDES_AGN["SNID"] = OzDES_AGN["SNID"].str.decode("utf-8")
    OzDES_AGN["AGN comment"] = OzDES_AGN["AGN comment"].str.decode("utf-8")
    OzDES_AGN["SNID"] = (
        OzDES_AGN["SNID"].map(lambda x: x.lstrip("SNID_")).astype(np.int32)
    )
    OzDES_AGN = OzDES_AGN.drop_duplicates()

    # Load human + automatic tagging (see M. Sako)
    various_flags = pd.read_csv(
        "./extra_lists/FlagsMerged_Sako.csv",
        comment="#",
        usecols=[
            "SNID",
            "PROB_BDT",
            "SCAN_MS1",
            "SCAN_MS2",
            "SCAN_MV",
            "COADD_OBJECTS_ID",
            "flagSNR3",
            "flagSNR5",
            "PROB_RF",
        ],
    )
    various_flags = various_flags.drop_duplicates()

    # Other SNe types from DES
    SLSN = pd.read_csv("./extra_lists/SLSN.csv", comment="#", delimiter=" ")
    Hounsell_CC = pd.read_csv("./extra_lists/Hounsell_templates_SNID.csv", comment="#")

    for method in method_list:
        lu.print_blue(method)
        dic_tag_SNIDs = {}
        dic_tag_stats = {}
        for tag in [
            "OzDES_AGN",
            "visual_nonSN_MS",
            "visual_nonSN_MV",
            "nonSN_RF",
            "nonSN_BDT",
            "nonSN_SNR3",
            "nonSN_SNR5",
            "HCC",
            "SLSN",
        ]:
            dic_tag_stats[tag] = []

        list_sets = (
            cu.list_seeds_set[0]
            if method == "S"
            else [k for k in cu.list_seeds_set.keys()]
        )
        if method == None or method == "specIa":
            list_sets = [0]  # hack to query this once
        for myset in list_sets:
            dic_tag_SNIDs[myset] = {}
            k = f"{method}_{myset}"
            if k in dic_photoIa_sel.keys() or method == None or method == "specIa":
                if method == None:
                    df = dic_photoIa_sel
                elif method == "specIa":
                    df = dic_photoIa_sel[
                        dic_photoIa_sel["SNTYPE"].isin(cu.spec_tags["Ia"])
                    ]
                else:
                    df = dic_photoIa_sel[k]
                df = pd.merge(
                    df, OzDES_AGN[["SNID", "qop", "AGN comment"]], on="SNID", how="left"
                )
                df = pd.merge(df, various_flags, on="SNID", how="left")

                # criteria
                dic_tag_SNIDs[myset]["OzDES_AGN"] = df[
                    df["AGN comment"].isin(["AGN", "Possible AGN"])
                ].SNID
                dic_tag_SNIDs[myset]["visual_nonSN_MS"] = df[
                    (df["SCAN_MS1"] == 0) | (df["SCAN_MS2"] == 0)
                ].SNID
                dic_tag_SNIDs[myset]["visual_nonSN_MV"] = df[(df["SCAN_MV"] == 0)].SNID
                dic_tag_SNIDs[myset]["nonSN_RF"] = df[(df["PROB_RF"] < 0.5)].SNID
                dic_tag_SNIDs[myset]["nonSN_BDT"] = df[(df["PROB_BDT"] < 0.5)].SNID
                dic_tag_SNIDs[myset]["nonSN_SNR3"] = df[(df["flagSNR3"] == 0)].SNID
                dic_tag_SNIDs[myset]["nonSN_SNR5"] = df[(df["flagSNR5"] == 0)].SNID

                dic_tag_SNIDs[myset]["HCC"] = df[
                    df.SNID.isin(Hounsell_CC.SNID.values)
                ].SNID
                dic_tag_SNIDs[myset]["SLSN"] = df[df.SNID.isin(SLSN.SNID.values)].SNID

                for tag in dic_tag_SNIDs[myset].keys():
                    dic_tag_stats[tag].append(len(dic_tag_SNIDs[myset][tag]))
        print(dic_tag_stats)

        if verbose:
            print(
                "OzDES_AGN",
                [
                    dic_tag_SNIDs[tt]["OzDES_AGN"].values.tolist()
                    for tt in list_sets
                    if len(dic_tag_SNIDs[tt]) > 0
                ],
            )
            print(
                "visual_nonSN_MV",
                [
                    dic_tag_SNIDs[tt]["visual_nonSN_MV"].values.tolist()
                    for tt in list_sets
                    if len(dic_tag_SNIDs[tt]) > 0
                ],
            )
        if method == "specIa":
            for k in dic_tag_SNIDs[0].keys():
                if (len(dic_tag_SNIDs[0][k]) < 10) & (len(dic_tag_SNIDs[0][k]) > 0):
                    print(
                        k, [dic_tag_SNIDs[0][k].values.tolist()],
                    )

    return dic_tag_SNIDs
