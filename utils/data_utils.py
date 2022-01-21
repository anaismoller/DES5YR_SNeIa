import re, glob, os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from astropy.table import Table
from utils import plot_utils as pu
from utils import logging_utils as lu
from utils import metric_utils as mu
from utils import conf_utils as cu
from utils import cuts as cuts


def print_stats(list_tmp, context="", nround=4):
    print(
        f"{context}: {str(np.round(np.mean(np.array(list_tmp).flatten()),nround))} pm {str(np.round(np.std(np.array(list_tmp).flatten()),nround))} max: {str(np.round(np.max(np.array(list_tmp).flatten()),nround))}"
    )


def read_header_fits(fname, drop_separators=False):
    """Load SNANA formatted header and cast it to a PANDAS dataframe
    Args:
        fname (str): path + name to HEAD.FITS file
        drop_separators (Boolean): if -777 are to be dropped
    Returns:
        (pandas.DataFrame) dataframe from PHOT.FITS file (with ID)
        (pandas.DataFrame) dataframe from HEAD.FITS file
    """

    # load header
    header = Table.read(fname, format="fits")
    df_header = header.to_pandas()
    df_header["SNID"] = df_header["SNID"].astype(np.int32).copy()

    return df_header


def load_headers(path_files):
    list_files = glob.glob(f"{path_files}/*HEAD.FITS*")
    df_list = []
    for fil in list_files:
        df_list.append(read_header_fits(fil, drop_separators=True))
    df = pd.concat(df_list)
    df["Dz_FINAL_HOSTSPEC"] = df["REDSHIFT_FINAL"] - df["HOSTGAL_SPECZ"]
    df["IAUC"] = df["IAUC"].str.decode("utf-8")

    print(f"Loaded {len(df)} light-curves metadata in {path_files}")
    return df


def read_fits(fname, drop_separators=False):
    """Load SNANA formatted data and cast it to a PANDAS dataframe
    Args:
        fname (str): path + name to PHOT.FITS file
        drop_separators (Boolean): if -777 are to be dropped
    Returns:
        (pandas.DataFrame) dataframe from PHOT.FITS file (with ID)
        (pandas.DataFrame) dataframe from HEAD.FITS file
    """

    # load photometry
    dat = Table.read(fname, format="fits")
    df_phot = dat.to_pandas()
    # failsafe
    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    # load header
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    df_header["SNID"] = df_header["SNID"].astype(np.int32).copy()

    # add SNID to phot for skimming
    arr_ID = np.zeros(len(df_phot), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df_phot["SNID"] = arr_ID

    if drop_separators:
        df_phot = df_phot[df_phot.MJD != -777.000]

    return df_phot


def load_photometry(path_files):
    """
    Load headers and photometry
    """
    list_files = glob.glob(f"{path_files}/*PHOT.FITS*")
    df_list = []
    for fname in list_files:
        df_list.append(read_fits(fname, drop_separators=True))
    df = pd.concat(df_list)

    print(f"Loaded {len(df_list)} photometry files in {path_files}")
    return df


def load_salt_fits(path_files):
    """
    Load headers and photometry
    """
    df = pd.read_csv(
        path_files, index_col=False, comment="#", delimiter=" ", skipinitialspace=True,
    )
    df = df.rename(columns={"CID": "SNID"})
    df = df.rename(columns={"TYPE": "SNTYPE"})
    df["SNTYPE"] = df["SNTYPE"].astype(int)

    print(f"Loaded {len(df)} SALT2 fits in {path_files}")
    return df


def read_metric_filesingle(f, suffix=None):
    """ Read a metric file

    Args:
        metric_file (filename): file to be read
        suffix (Optional): if suffix to be added to model_name

    Returns:
        df (Pandas DataFrame): formatted metric df

    """
    tmp = pd.read_pickle(f)

    # hack to have more HP without changing SNN name formatting
    if "62x" in tmp.model_name.values[0]:
        tmp["model_name"] = tmp["model_name"].apply(lambda x: f"{x}_extrashuffle")
    elif "63x" in tmp.model_name.values[0]:
        tmp["model_name"] = tmp["model_name"].apply(lambda x: f"{x}_cyclicphases204060")

    if suffix:
        tmp["model_name"] = tmp["model_name"].apply(lambda x: f"{x}_{suffix}")

    # reduced model name
    tmp["reduced_model_name"] = tmp["model_name"].apply(
        lambda x: x.replace("vanilla_S_0_CLF_2_R_zspe_photometry_DF_1.0_N_", "")
        if "DF_1.0" in x
        else x.replace("vanilla_S_0_CLF_2_R_zspe_photometry_DF_0.2_N_", "")
    )
    return tmp


def read_metric_files(metric_files, suffix=None):
    """ Read and concatenate all metric files

    Args:
        metric_files (list): list of files to be read and concat
        suffix (Optional): if suffix to be added to model_name

    Returns:
        df (Pandas DataFrame): concatenated  files
    """

    if len(metric_files) > 2:
        list_df = []
        for f in metric_files:
            tmp = read_metric_filesingle(f, suffix=suffix)
            list_df.append(tmp)
        df = pd.concat(list_df)
    elif len(metric_files) == 1:
        df = read_metric_filesingle(metric_files[0], suffix=suffix)
    else:
        lu.print_red("No metric files to be read")
        raise AttributeError

    return df


def get_metric_singleseed_files(path_in, arg_name, model_name=None):
    """ Get list of metric files

    Args:
        path_in (Path): path to metrics files
        arg_name (str): name on argument
        model_name (str): if only a single model name to be processed

    Returns:
        metric_files_singleseed (list): list of files with Seed 0
    """
    if model_name:
        metric_files_singleseed = glob.glob(f"{path_in}/{model_name}/METRICS*")
    else:
        metric_files_singleseed = glob.glob(f"{path_in}/*S_0_*/METRICS*")

    if len(metric_files_singleseed) < 1:
        lu.print_red(
            f"Please provide correct {arg_name} path, currently {path_in}/{model_name}/METRICS*"
        )
        raise AttributeError

    return metric_files_singleseed


def get_stats_cal(metric_files_singleseed, path_dump, description):
    mean_acc, std_acc, _ = get_mean_stats(
        metric_files_singleseed,
        what="all_accuracy",
        output_path=f"{path_dump}/SNN_{description}_accuracy.txt",
    )

    lu.print_green(f"{description}")
    for k in np.sort(list(mean_acc.keys())):
        print(k, f"accuracy: {mean_acc[k]} \\pm {std_acc[k]}")

    # Calibration
    mean_bins, std_bins, TPF = get_mean_stats(
        metric_files_singleseed, what="calibration"
    )
    pu.plot_calibration(
        mean_bins,
        std_bins,
        TPF,
        f"{path_dump}/plots/SNN_{description}_calibration.png",
    )


def get_mean_stats(metric_files_singleseed, what="calibration", output_path=None):
    """ Get stats from metrics for set 0 seeds

    Args:
        metric_files_singleseed (list): list of files with Seed 0
        what (str): which stat to compute 'calibration' or 'accuracy'
        output_path (Filename+Path): File to be saved stats
    """

    mean_bins = {}
    std_bins = {}
    TPF = {}
    for fname in metric_files_singleseed:
        norm_name = Path(fname).stem.split("DF_1.0_N_")[-1].split("_lstm")[0]

        list_df = []
        for seed in cu.list_seeds_set[0]:
            search_query = re.sub(r"S\_\d+_", f"S_{seed}_", fname)
            f = glob.glob(search_query)[0]
            tmp = pd.read_pickle(f)
            list_df.append(tmp)

        df = pd.concat(list_df)
        if what == "calibration":
            mean_bins[norm_name] = df["calibration_mean_bins"].values.mean()
            std_bins[norm_name] = df["calibration_mean_bins"].values.std()
            TPF[norm_name] = df["calibration_TPF"].mean()
        elif "accuracy" in what:
            mean_bins[norm_name] = round(df[what].values.mean(), 2)
            std_bins[norm_name] = round(df[what].values.std(), 2)
            if std_bins[norm_name] == 0.0:
                mean_bins[norm_name] = round(df[what].values.mean(), 3)
                std_bins[norm_name] = round(df[what].values.std(), 3)
            TPF[norm_name] = [0]
        else:
            lu.print_red(
                f"Please provide correct stat to compute ['calibration','accuracy']"
            )
            raise AttributeError

    if output_path:
        text_file = open(output_path, "w")
        for k in mean_bins.keys():
            text_file.write(f"{k}: {mean_bins[k]} \\pm {std_bins[k]} \n")
        text_file.close()

    return mean_bins, std_bins, TPF


def load_fitfile(path_fits):
    """Load the FITOPT file as a pandas dataframe

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (pandas.DataFrame) dataframe with FITOPT data
    """

    if (
        Path(f"{path_fits}/FITOPT000.FITRES").exists()
        or Path(f"{path_fits}/FITOPT000.FITRES.gz").exists()
    ):
        fit_name = (
            f"{path_fits}/FITOPT000.FITRES"
            if Path(f"{path_fits}/FITOPT000.FITRES").exists()
            else f"{path_fits}/FITOPT000.FITRES.gz"
        )

    elif Path(f"{path_fits}").exists():
        fit_name = path_fits

    else:
        lu.print_red("Not a valid FITS name/path", path_fits)

    df = pd.read_csv(
        fit_name, index_col=False, comment="#", delimiter=" ", skipinitialspace=True
    )

    # Rename CID to SNID
    # SNID is CID in FITOPT000.FITRES
    df = df.rename(columns={"CID": "SNID"})
    df["SNID"] = df["SNID"].astype("int64")

    return df


def load_pred(pred_file, suffix=None):

    tmp = pd.read_pickle(pred_file)
    if "variational" or "bayesian" in pred_file:
        tmp = tmp.rename(
            columns={
                "all_class0_median": "all_class0",
                "all_class1_median": "all_class1",
            }
        )
        tmp = tmp.reset_index(drop=True)
    df = tmp[["SNID", "all_class0", "all_class1", "target"]]
    df.loc[:, "SNID"] = df["SNID"].astype(np.int64).values

    # get predicted class
    key_pred_targ = (
        "predicted_target" if suffix == None else f"predicted_target_{suffix}"
    )
    df[key_pred_targ] = (
        df[["all_class0", "all_class1"]].idxmax(axis=1).str.strip("class_").astype(int)
    )
    if suffix != None:
        df = df.rename(
            {
                "all_class0": f"all_class0_{suffix}",
                "all_class1": f"all_class1_{suffix}",
            },
            axis="columns",
        )

    return df


def load_preds(path_p, df_metadata, prob_key="all_class0"):
    """
    Load predictions and format
    """
    if os.path.exists(path_p):
        if Path(path_p).suffix == ".csv":
            df_preds_tmp = pd.read_csv(path_p)
        elif Path(path_p).suffix == ".pickle":
            df_preds_tmp = pd.read_pickle(path_p)
        else:
            lu.print_red("Format predictions not supported", path_p)
        df_preds_tmp["SNID"] = df_preds_tmp["SNID"].astype(np.int32)
        df_preds_tmp["all_class0"] = df_preds_tmp[prob_key]

        # only check those lcs that pass cuts
        df_preds = pd.DataFrame()
        df_preds = pd.merge(df_metadata, df_preds_tmp, on="SNID", how="left")

        return df_preds


def load_preds_addsuffix(path_p, prob_key="all_class0", suffix=None):
    """
    Load predictions and add suffix
    """
    if os.path.exists(path_p):
        if Path(path_p).suffix == ".csv":
            df_preds_tmp = pd.read_csv(path_p)
        elif Path(path_p).suffix == ".pickle":
            df_preds_tmp = pd.read_pickle(path_p)
        else:
            lu.print_red("Format predictions not supported", path_p)
        df_preds_tmp["SNID"] = df_preds_tmp["SNID"].astype(np.int32)

        prefix = prob_key.split("0")[0]

        if prob_key == "all_class0":
            to_max = 0
        else:
            df_preds_tmp = df_preds_tmp.rename(
                {prob_key: "all_class0", prob_key.replace("0", "1"): "all_class1",},
                axis="columns",
            )
            df_preds_tmp = df_preds_tmp.reset_index(drop=True)

        # get predicted class
        key_pred_targ = (
            "predicted_target" if suffix == None else f"predicted_target_{suffix}"
        )

        df_preds_tmp[key_pred_targ] = (
            df_preds_tmp[["all_class0", "all_class1"]]
            .idxmax(axis=1)
            .str.strip("class_")
            .astype(int)
        )
        df_preds_tmp = df_preds_tmp.rename(
            {f"{prefix}0": f"{prefix}0_{suffix}", f"{prefix}1": f"{prefix}1_{suffix}",},
            axis="columns",
        )

        # keep only certain columns
        df_preds = df_preds_tmp[
            [
                "SNID",
                "target",
                f"{prefix}0_{suffix}",
                f"{prefix}1_{suffix}",
                f"predicted_target_{suffix}",
            ]
        ]
        # add std if Bayesian
        if "median" in prob_key:
            df_preds[f"{prefix}0_std_{suffix}"] = (
                df_preds_tmp[f"{prefix}0_std"].copy().values
            )
            df_preds[f"{prefix}1_std_{suffix}"] = (
                df_preds_tmp[f"{prefix}1_std"].copy().values
            )
        return df_preds


def add_ensemble_methods(df_dic_preds, norm):
    """ Add columns for targets using ensemble methods
    Args:
        df_dic_preds (DataFrame): 
        norm (str): norm to process in dictionary
    Returns:

    """
    list_sets = []
    for set_model_average in cu.list_sets:
        list_seed_ensemble = cu.list_seeds_set[set_model_average]
        # target ensemble
        list_pred_targets_ensemble = [
            f"predicted_target_S_{k}"
            for k in list_seed_ensemble
            if f"predicted_target_S_{k}" in df_dic_preds[norm].keys()
        ]
        if len(list_pred_targets_ensemble) > 0:
            df_dic_preds[norm][
                f"average_targets_set_{set_model_average}"
            ] = df_dic_preds[norm][list_pred_targets_ensemble].sum(axis=1) / len(
                list_pred_targets_ensemble
            )

            # note that here is a target threshold so average target>0.5 is target 1
            df_dic_preds[norm][
                f"predicted_target_average_target_set_{set_model_average}"
            ] = df_dic_preds[norm][f"average_targets_set_{set_model_average}"].apply(
                lambda x: 1 if x > 0.5 else 0
            )

            # model averaging in probabilities with threshold 0.5
            list_pred_probs_ensemble = [
                f"all_class0_S_{k}"
                for k in list_seed_ensemble
                if f"all_class0_S_{k}" in df_dic_preds[norm].keys()
            ]

            df_dic_preds[norm][
                f"average_probability_set_{set_model_average}"
            ] = df_dic_preds[norm][list_pred_probs_ensemble].sum(axis=1) / len(
                list_pred_probs_ensemble
            )
            # note that here is a probability threshold so prob>0.5 is target 0
            df_dic_preds[norm][
                f"predicted_target_average_probability_set_{set_model_average}"
            ] = df_dic_preds[norm][
                f"average_probability_set_{set_model_average}"
            ].apply(
                lambda x: 0 if x > 0.5 else 1
            )

            # save sets list
            list_sets.append(set_model_average)

    return df_dic_preds, list_sets


def get_preds_seeds_merge(
    my_seed_list,
    path,
    norm="cosmo",
    model_prefix="",
    model_suffix="_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean",
):
    tmp_pred_dic = {}
    missing_seeds = []
    for seed in my_seed_list:
        if model_prefix == "variational" or model_prefix == "bayesian":
            pred_file_name = (
                f"{path}/{model_prefix}*S_{seed}{model_suffix}/PRED*_aggregated.pickle"
            )
        else:
            pred_file_name = f"{path}/{model_prefix}*S_{seed}{model_suffix}/PRED*"
        list_pred_file = glob.glob(pred_file_name)
        if len(list_pred_file) > 0:
            pred_file = list_pred_file[0]
            tmp_pred_dic[f"{norm}_S_{seed}"] = load_pred(pred_file, suffix=f"S_{seed}")
        else:
            lu.print_red(f"Missing seed {seed} : {pred_file_name}")
            missing_seeds.append(seed)

    # merge to combine predictions
    list_by_norm = [
        f"{norm}_S_{seed}" for seed in my_seed_list if seed not in missing_seeds
    ]
    try:
        dfList = [tmp_pred_dic[k] for k in list_by_norm]
    except Exception:
        import ipdb

        ipdb.set_trace()
    df_dic = reduce(lambda df1, df2: pd.merge(df1, df2, on=["SNID", "target"],), dfList)

    return df_dic


def get_norm(hist_data, hist_sim, err_data, method="likelihood"):

    # 1. brute force
    if method == "brute":
        norm = ratio.max()

    # 2.solving a linear equation
    elif method == "lstsq":
        A = np.vstack([hist_data, np.ones(len(hist_data))]).T
        y = hist_sim
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        norm = 1 / m

    elif method == "likelihood":
        # std likelihood
        # from http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/
        from scipy import optimize

        def squared_loss(theta, x=hist_data, y=hist_sim, e=err_data):
            dy = y - theta[0] - theta[1] * x
            return np.sum(0.5 * (dy / e) ** 2)

        theta1 = optimize.fmin(squared_loss, [0, 0], disp=False)
        norm = 1 / theta1[1]

    elif method == "bayesian":
        # bayesian
        # modified from http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/
        x = hist_data
        y = hist_sim
        e = err_data

        def log_prior(theta, x, y):
            # 1. intercept must be around 0
            # np.allclose(theta[0],[0.0])
            # 2. my ratio y/mx can't go beyond 1 (efficiency limit)
            # all(y<=theta[1]*x)
            if np.allclose(theta[0], [0.0]) and np.all(x * theta[1] < y):
                return 0
            else:
                return -np.inf  # recall log(0) = -inf

        def log_likelihood(theta, x, y, e, sigma_B):
            dy = y - theta[0] - theta[1] * x

            logL1 = -0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
            logL2 = (
                np.log(1)
                - 0.5 * np.log(2 * np.pi * sigma_B ** 2)
                - 0.5 * (dy / sigma_B) ** 2
            )
            return np.sum(np.logaddexp(logL1, logL2))

        def log_posterior(theta, x, y, e, sigma_B):
            return log_prior(theta, x, y) + log_likelihood(theta, x, y, e, sigma_B)

        # Note that this step will take a few minutes to run!
        ndim = 2  # number of parameters in the model
        nwalkers = 50  # number of MCMC walkers
        nburn = 100  # "burn-in" period to let chains stabilize
        nsteps = 1500  # number of MCMC steps to take

        # set theta near the maximum likelihood, with
        np.random.seed(0)
        starting_guesses = np.zeros((nwalkers, ndim))
        starting_guesses[:, :2] = np.random.normal(theta1, 1, (nwalkers, 2))
        import emcee

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, args=[x, y, e, 50]
        )
        sampler.run_mcmc(starting_guesses, nsteps)

        sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
        sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)

        theta3 = np.mean(sample[:, :2], 0)
        norm = 1 / theta3[1]

        lu.print_red("Norm ratio: using Gaussian Likelihood,need to change to Poisson")

    return norm


def data_sim_ratio(
    data, sim, var="HOST_MAG_i", min_var=15, path_plots=None, suffix=None, norm=None
):
    """
    Ratio between data and simulation in a given variable
    """
    # Init
    # TODO: no hardcut for lower limit
    data_var = data[data[var] > min_var][var]
    sim_var = sim[sim[var] > min_var][var]

    minv = min([x.quantile(0.01) for x in [data_var, sim_var]])
    maxv = max([x.quantile(0.99) for x in [data_var, sim_var]])
    bins = np.linspace(minv, maxv, 12)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    hist_data, _ = np.histogram(data_var, bins=bins)
    hist_sim, _ = np.histogram(sim_var, bins=bins)
    err_data = np.sqrt(hist_data)
    err_sim = np.sqrt(hist_sim)

    ratio = hist_data / hist_sim

    if norm != None:
        norm = norm
    else:
        norm = get_norm(hist_data, hist_sim, err_data)

    ratio = ratio / norm
    err_ratio = np.sqrt((err_data / hist_data) ** 2 + (err_sim / hist_sim) ** 2) * ratio
    err_ratio = np.nan_to_num(err_ratio)

    # ratio for spec Ias
    # using previously computed norm
    data_var_spec = data[(data[var] > min_var) & (data["SNTYPE"].isin([1, 101]))][var]
    hist_data_spec, _ = np.histogram(data_var_spec, bins=bins)
    err_data_spec = np.sqrt(hist_data_spec)
    ratio_spec = hist_data_spec / (hist_sim * norm)
    err_ratio_spec = (
        np.sqrt((err_data_spec / hist_data_spec) ** 2 + (err_sim / hist_sim) ** 2)
        * ratio_spec
    )
    err_ratio_spec = np.nan_to_num(err_ratio_spec)

    # save as dataframe
    df = pd.DataFrame()
    df["x"] = bin_centers
    df["ratio"] = ratio
    df["err_ratio"] = err_ratio
    df["ndata"] = hist_data
    df["nsim"] = hist_sim
    df["ratio_spec"] = ratio_spec
    df["err_ratio_spec"] = err_ratio_spec
    df["ndata_spec"] = hist_data_spec
    df.ratio_variable = var

    if path_plots:
        import matplotlib.pyplot as plt

        # plot histos
        fig = plt.figure()
        plt.errorbar(
            bin_centers,
            hist_sim * norm,
            yerr=np.sqrt(hist_sim * norm),
            label="normed sim",
            fmt="o",
            color=pu.color_dic["sim"],
        )
        plt.errorbar(
            bin_centers,
            hist_data_spec,
            yerr=err_data_spec,
            label="spectroscopic SNe Ia",
            fmt="o",
            color=pu.color_dic["spec"],
        )
        plt.errorbar(
            bin_centers,
            hist_data,
            yerr=err_data,
            label="photometric SNe Ia",
            fmt="o",
            color=pu.color_dic["data"],
        )
        plt.legend()
        plt.xlabel(var)
        plt.ylabel("# SNe Ia")
        nameout = (
            f"{path_plots}/hist_data_sim_{var}_{suffix}.png"
            if suffix != None
            else f"{path_plots}/hist_data_sim_{var}.png"
        )
        plt.savefig(nameout)
        plt.close(fig)

        # plot ratio
        fig = plt.figure()
        plt.errorbar(
            bin_centers,
            ratio,
            yerr=err_ratio,
            fmt="o",
            color=pu.color_dic["ratio"],
            label="ratio photo SN Ia",
        )
        plt.errorbar(
            bin_centers,
            ratio_spec,
            yerr=err_ratio_spec,
            fmt="o",
            color=pu.color_dic["spec"],
            label="ratio spec SN Ia",
        )
        plt.xlabel(var)
        plt.ylabel("photometric SNe Ia efficiency")
        plt.legend()
        nameout = (
            f"{path_plots}/ratio_data_sim_{var}_{suffix}.png"
            if suffix != None
            else f"{path_plots}/ratio_data_sim_{var}.png"
        )
        plt.savefig(nameout)
        plt.close(fig)

    return df, minv, maxv


def add_uncertainties_BNN(df_dic_preds, norm):
    """
    Ensemble methods uncertainties
    Average_probability_set_std: std between average probability (as ensemble uncertainty for a given classification)

    """
    df_dic_tmp = df_dic_preds

    for set_model_average in cu.list_sets:
        list_seed_ensemble = cu.list_seeds_set[set_model_average]
        # 1. average uncertainties in a given set
        # setting covariance between models to zero
        # see wikipedia std for formula
        list_unc_ensemble = [
            f"all_class0_std_S_{k}"
            for k in list_seed_ensemble
            if f"all_class0_std_S_{k}" in df_dic_preds[norm].keys()
        ]

        df_dic_tmp[norm][
            f"average_probability_set_{set_model_average}_meanstd"
        ] = np.sqrt(
            (df_dic_tmp[norm][list_unc_ensemble] ** 2).sum(axis=1)
            / len(list_unc_ensemble)
        )

        # 2. std average probability
        list_probs_ensemble = [
            f"all_class0_S_{k}"
            for k in list_seed_ensemble
            if f"all_class0_S_{k}" in df_dic_preds[norm].keys()
        ]

        df_dic_tmp[norm][
            f"average_probability_set_{set_model_average}_stdprob"
        ] = df_dic_tmp[norm][list_probs_ensemble].std(axis=1)

    return df_dic_tmp


def load_merge_all_preds(
    path_class="./",
    model_name="vanilla_S_*_zspe*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
    norm="cosmo_quantile",
    prob_key="all_class0",
):
    # load sim predictions
    if "median" in prob_key:
        suffix = "aggregated"
    else:
        suffix = ""
    cmd = f"{path_class}/{model_name}/PRED_*{suffix}.pickle"
    list_path_p = glob.glob(cmd)

    print(f"Loading {len(list_path_p)} files")
    # SNIDs are the same for all preds
    list_df_preds = []
    preds = {}
    for path_p in list_path_p:
        seed = re.search(r"(?<=S\_)\d+", path_p).group()
        list_df_preds.append(
            load_preds_addsuffix(path_p, prob_key=prob_key, suffix=f"S_{seed}")
        )
    # merge all predictions
    if len(list_df_preds) > 0:
        preds[norm] = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=["SNID", "target"],), list_df_preds
        )
        # ensemble methods + metadata
        preds, list_sets = add_ensemble_methods(preds, norm)
        print("Predictions for", list_sets)

        if "median" in prob_key:
            # add uncertainties and mean uncertainties
            preds = add_uncertainties_BNN(preds, norm)
    else:
        lu.print_red("No predictions here")
    return preds


def get_sample_stats(
    df_sel, suffix="", methods=["single_model", "average_probability"],
):
    """
    Mean and std of samples with photo/specc subs
    """
    df_photoIa_stats = pd.DataFrame()
    for method in methods:
        arr = []
        if method == "single_model":
            for seed in cu.list_seeds_set[0]:
                arr.append(cuts.photo_sel_prob(df_sel, prob_key=f"all_class0_S_{seed}"))
        else:
            for modelset in cu.list_sets:
                target_key = f"predicted_target_{method}_set_{modelset}"
                if target_key in df_sel.keys():
                    arr.append(cuts.photo_sel_target(df_sel, target_key=target_key,))
                else:
                    lu.print_red(f"Set {modelset} not available for stats")
        photoIa, specIa, specCC, specother = cuts.do_arr_stats(arr)
        dic_tmp = {
            "norm": "cosmo_quantile",
            "method": method.replace("_", " "),
            f"photoIa{suffix}": photoIa,
            f"specIa{suffix}": specIa,
            f"specCC{suffix}": specCC,
            f"specOther{suffix}": specother,
        }
        df_photoIa_stats = df_photoIa_stats.append(dic_tmp, ignore_index=True)
    return df_photoIa_stats
