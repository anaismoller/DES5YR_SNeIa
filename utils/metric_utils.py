import numpy as np
import pandas as pd
from sklearn import metrics
from . import logging_utils as lu


def performance_metrics(
    df,
    sample_target=0,
    key_pred_targ="predicted_target",
    compute_auc=True,
    print_metrics=False,
):
    """Get performance metrics
    AUC: only valid for binomial classification, input proba of highest label class.

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, class1]
        (optional) sample_target (str): for SNIa sample default is target 0
        key_pred_targ (str): column name of predicted target to be used
        print_metrics (Boolean): if metrics to be printed

    Returns:
        accuracy, auc, purity, efficiency,truepositivefraction
    """
    n_targets = len(np.unique(df["target"]))

    # Accuracy & AUC
    balancedaccuracy = metrics.balanced_accuracy_score(
        df["target"].astype(int), df[key_pred_targ].astype(int)
    )
    balancedaccuracy = balancedaccuracy * 100

    accuracy = metrics.accuracy_score(
        df["target"].astype(int), df[key_pred_targ].astype(int)
    )
    accuracy = accuracy * 100
    if n_targets == 2 and compute_auc:  # valid for biclass only
        auc = round(metrics.roc_auc_score(df["target"], df["all_class1"]), 4)
    else:
        auc = 0.0

    SNe_Ia = df[df["target"] == sample_target]
    SNe_CC = df[df["target"] != sample_target]
    TP = len(SNe_Ia[SNe_Ia[key_pred_targ] == sample_target])
    FP = len(SNe_CC[SNe_CC[key_pred_targ] == sample_target])

    P = len(SNe_Ia)
    N = len(SNe_CC)

    truepositivefraction = P / (P + N) if (P + N) > 0 else 0
    purity = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    efficiency = 100.0 * TP / P if P > 0 else 0

    if print_metrics:
        print(
            f"Accuracy: {balancedaccuracy} | Purity: {purity} | Efficiency {efficiency}"
        )

    return accuracy, balancedaccuracy, auc, purity, efficiency, truepositivefraction


def txt_meanstd_w_significant_figures(arr):
    # default 2
    decimals = 2
    if round(np.array(arr).std(), decimals) == 0.0:
        decimals = 3
        if round(np.array(arr).std(), 3) == 0.0:
            decimals = 4
    txt = f"${round(np.array(arr).mean(),decimals)} pm {round(np.array(arr).std(),decimals)}$"

    return txt


def get_multiseed_performance_metrics(
    df,
    key_pred_targ_prefix="predicted_target_S_",
    list_seeds=[0],
    df_txt=None,
    dic_prefilled_keywords=None,
):
    """
    """
    # in case a seed is missing
    list_seeds_updated = [
        k for k in list_seeds if f"{key_pred_targ_prefix}{k}" in df.keys()
    ]
    if not set(list_seeds) == set(list_seeds_updated):
        lu.print_red(
            "Missing seed", [k for k in list_seeds if k not in list_seeds_updated]
        )
        lu.print_red("processing with missing seed", dic_prefilled_keywords)
        list_seeds = list_seeds_updated

    balacc_dic = []
    acc_dic = []
    eff_dic = []
    pur_dic = []
    for seed in list_seeds:
        (
            accuracy,
            balancedaccuracy,
            auc,
            purity,
            efficiency,
            truepositivefraction,
        ) = performance_metrics(
            df, key_pred_targ=f"{key_pred_targ_prefix}{seed}", compute_auc=False,
        )
        balacc_dic.append(balancedaccuracy)
        acc_dic.append(accuracy)
        eff_dic.append(efficiency)
        pur_dic.append(purity)

    # increase decimals in case missing significant figures in accuracy
    txt_acc = txt_meanstd_w_significant_figures(np.array(acc_dic))
    txt_balacc = txt_meanstd_w_significant_figures(np.array(balacc_dic))
    txt_eff = txt_meanstd_w_significant_figures(np.array(eff_dic))
    txt_pur = txt_meanstd_w_significant_figures(np.array(pur_dic))

    if isinstance(df_txt, pd.DataFrame):
        dic_prefilled_keywords["notbalanced_accuracy"] = txt_acc
        dic_prefilled_keywords["accuracy"] = txt_balacc
        dic_prefilled_keywords["efficiency"] = txt_eff
        dic_prefilled_keywords["purity"] = txt_pur
        df_txt = df_txt.append(dic_prefilled_keywords, ignore_index=True)
        return df_txt
    else:
        print(
            f"Balanced Accuracy: {txt_balacc} Notbalanced accuracy: {txt_acc} Efficiency: {txt_eff} Purity: {txt_pur} "
        )


def reformatting_tolatex(df, norm_list=["cosmo", "cosmo_quantile"], dataset="balanced"):
    tmp = {}
    for norm in norm_list:
        tmp = df[(df.dataset == dataset) & (df.norm == norm)][
            ["method", "accuracy", "efficiency", "purity"]
        ]
        print("\multicolumn", {tmp.shape[-1]}, "{c}{", norm, "}\\\\")
        print("\hline")
        print(tmp.to_latex(index=False))
