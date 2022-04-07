import logging
import argparse
import ipdb
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

from utils import cuts as cuts
from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import logging_utils as lu
from utils import science_utils as su

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
        "--path_data",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits",
        help="Path to data",
    )
    parser.add_argument(
        "--path_fits",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_JLA_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits",
    )
    parser.add_argument(
        "--path_class",
        type=str,
        default=f"{DES}/DES5YR/data_preds/snndump_26XBOOSTEDDES/models/",
        help="Path to data predictions",
    )

    parser.add_argument(
        "--path_dump", default=f"./dump_DES5YR", type=str, help="Path to output",
    )

    # Init
    args = parser.parse_args()
    path_data = args.path_data
    path_class = args.path_class
    path_fits = args.path_fits
    path_dump = args.path_dump
    os.makedirs(path_dump, exist_ok=True)
    path_plots = f"{path_dump}/plots_sample_wz/"
    os.makedirs(path_plots, exist_ok=True)
    path_samples = f"{path_dump}/samples/"
    os.makedirs(path_samples, exist_ok=True)

    # logger
    logger = setup_logging()

    # save configuration
    logger.info("_______________")
    logger.info("CONFIGURATION")
    logger.info(
        f"python paper.py --path_dump {path_dump} --path_data {path_data}  --path_fits {path_fits} --path_class {path_class} "  # --path_sims {path_sims}"
    )

    # PHOTOMETRY QUALITY
    # performed directly in SuperNNova
    # for consistency we compute the stats here as well
    # cuts.data_processing("DATA PROCESSING", path_data, logger)

    logger.info("_______________")
    logger.info("PHOTOMETRIC SNE IA WITH HOST SPECTROSCOPIC REDSHIFTS")

    logger.info("SELECTION CUTS")
    # Load all metadata
    df_metadata = du.load_headers(path_data)

    # stats
    print(f"All DES candidates {len(df_metadata)}")
    print(f"# events with spec z: {len(df_metadata[df_metadata['REDSHIFT_FINAL']>0])}")
    print(f"# events with hosts: {len(df_metadata[df_metadata['HOSTGAL_MAG_r']<40])}")
    print(
        f"# events with hosts mag_r<24: {len(df_metadata[df_metadata['HOSTGAL_MAG_r']<24])}"
    )
    cuts.spec_subsamples(df_metadata, logger)

    # OOD cuts
    # detection cuts (fast + multi-season)
    df_metadata_w_dets = cuts.detections("DETECTIONS", df_metadata, logger)
    # multiseason
    df_metadata_w_multiseason = cuts.transient_status(
        "MULTI-SEASON", df_metadata_w_dets, logger
    )

    logger.info("")
    logger.info("POSSIBLE CONTAMINATION")
    # possible AGNs from lists
    dic_tag_SNIDs = cuts.stats_possible_contaminants(
        df_metadata_w_multiseason, method_list=["specIa", None], verbose=False
    )

    # REDSHIFT
    df_metadata_sel = cuts.redshift("REDSHIFT", df_metadata_w_multiseason, logger)
    SNID_3yr_spec = df_metadata_sel[
        (df_metadata_sel["IAUC"].str.contains("DES15|DES14|DES13"))
        & (df_metadata_sel["SNTYPE"].isin(cu.spec_tags["Ia"]))
    ]["SNID"].values

    # SALT converges
    df_salt = du.load_salt_fits(path_fits)
    df_metadata_sel = cuts.salt_basic(
        "SALT sampling + convergence", df_metadata_sel, df_salt, logger
    )
    SNID_3yr_spec_salt = df_metadata_sel[
        (df_metadata_sel["IAUC"].str.contains("DES15|DES14|DES13"))
        & (df_metadata_sel["SNTYPE"].isin(cu.spec_tags["Ia"]))
    ]["SNID"].values

    lu.print_red(
        "Missing specIa after SALT fit (no cut, just fit)",
        len(SNID_3yr_spec) - len(SNID_3yr_spec_salt),
    )
    print("SNIDs", [k for k in SNID_3yr_spec if k not in SNID_3yr_spec_salt])

    # save for later
    df_metadata_selection_cuts = df_metadata_sel
    df_metadata_selection_cuts.to_csv(f"{path_samples}/only_selection_cuts.csv")

    specIas = df_metadata_sel[df_metadata_sel["SNTYPE"].isin(cu.spec_tags["Ia"])]

    # logger.info("")
    # cuts.check_peak("CHECK PEAK", df_metadata_sel, path_class, logger)

    # dump pass qual cuts
    # df_metadata_sel['SNID'].to_csv('SNIDs_passqualcuts.csv')

    logger.info("")
    logger.info("PHOTOMETRIC CLASSIFICATION")

    # photometric samples with different selection methods
    # dic_df_photoIa: dictionary for each norm and classification method
    # df_photoIa_stats: table to fill for paper in next step

    # main method
    norm = "cosmo_quantile"
    method = "average_probability_set_0"

    dic_df_photoIa_wsalt, df_photoIa_stats = cuts.photo_norm(
        df_metadata_sel, path_class, path_dump, logger, path_plots=path_plots
    )

    # save sample with loose sel cuts
    dic_df_photoIa_wsalt[norm][method].to_csv(
        f"{path_samples}/BaselineDESsample_looseselcuts.csv"
    )

    # save lost spec Ias
    lost_spec = [
        k
        for k in specIas.SNID.values
        if k not in dic_df_photoIa_wsalt[norm][method].SNID.values
    ]
    df_spec = pd.DataFrame()
    df_spec["SNID"] = lost_spec
    df_spec.to_csv(f"{path_samples}/lost_specIa_selcuts_{norm}_{method}.csv")

    # add SALT2 and PRIMUS cuts
    dic_photoIa_sel = cuts.towards_cosmo(dic_df_photoIa_wsalt, df_photoIa_stats, logger)

    photo_Ia = dic_photoIa_sel[norm][method]

    photo_Ia.to_csv(f"{path_samples}/photoIa_{norm}_{method}.csv")
    photo_Ia.to_csv(f"{path_samples}/BaselineDESsample_JLAlikecuts.csv")

    lu.print_green(
        "Cosmo quantile + average_probability_set_0 photometric selection after ALL quality cuts",
        len(photo_Ia),
    )
    lu.print_green(
        f"PhotoIawz set 0 redshift range {photo_Ia.zHD.min()}-{photo_Ia.zHD.max()}"
    )

    specIas_JLA = su.apply_JLA_cut(specIas)
    pu.plot_mosaic_histograms_listdf(
        [photo_Ia, specIas_JLA],
        list_labels=["photo SNe Ia", "spec SNe Ia"],
        path_plots=path_plots,
        suffix="photospecJLA",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
        chi_bins=False,
    )

    lost_spec = [k for k in specIas_JLA.SNID.values if k not in photo_Ia.SNID.values]
    lu.print_green(
        f"Lost specIas during photometric classification + JLA {len(lost_spec)}"
    )
    pu.plot_mosaic_histograms_listdf(
        [specIas[specIas.SNID.isin(lost_spec)], specIas,],
        ["lost spec", "spec"],
        path_plots=path_plots,
        suffix="_specIas",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
        log_scale=True,
    )
    # save lost spec Ias
    df_spec = pd.DataFrame()
    df_spec["SNID"] = lost_spec
    df_spec.to_csv(f"{path_samples}/lost_specIa_{norm}_{method}_JLA.csv")

    lu.print_green(f"Overlap between methods")
    photo_Ia_S_0 = dic_photoIa_sel["cosmo_quantile"]["S_0"]
    photo_Ia_c = dic_photoIa_sel["cosmo"]["S_0"]
    dic_venn = {
        "photoIa w z cq set 0": set(photo_Ia.SNID.values),
        "photoIa w z cq S 0": set(photo_Ia_S_0.SNID.values),
        "photoIa w z c set 0": set(photo_Ia_c.SNID.values),
    }
    pu.plot_venn_percentages(dic_venn, path_plots=path_plots, data=True)

    photo_Ia_S_0.to_csv(f"{path_samples}/photoIa_cosmo_quantile_S_0.csv")

    print(
        f"overlap cosmo/cosmo quantile in prob average method",
        round(
            100
            * len(set(photo_Ia.SNID.values) & set(photo_Ia_c.SNID.values))
            / len(photo_Ia),
            2,
        ),
    )
    print(
        f"overlap cosmo quantile in single seed/prob average method",
        round(
            100
            * len(set(photo_Ia.SNID.values) & set(photo_Ia_S_0.SNID.values))
            / len(photo_Ia),
            2,
        ),
    )

    logger.info("")
    logger.info("POSSIBLE CONTAMINATION")
    # possible AGNs from lists
    dic_tag_SNIDs = cuts.stats_possible_contaminants(dic_photoIa_sel[norm])

    lu.print_green("IF WE ONLY DID CUTS FOR CLASSIFICATION (after selection cuts)")
    # Load PRIMUS redshift info
    hostz_info = pd.read_csv("extra_lists/SNGALS_DLR_RANK1_INFO.csv")
    hostz_info["SPECZ_CATALOG"] = hostz_info.SPECZ_CATALOG.apply(
        lambda x: x[2:-1].strip(" ")
    )
    SNID_to_keep = hostz_info[hostz_info.SPECZ_CATALOG != "PRIMUS"].SNID.values.tolist()
    sel = df_metadata_sel[df_metadata_sel.SNID.isin(SNID_to_keep)]
    print(f"PRIMUS selection reduces {len(df_metadata_sel)} to {len(sel)}")
    sel_JLA = su.apply_JLA_cut(sel)
    print(f"JLA cuts reduces to {len(sel_JLA)}")
