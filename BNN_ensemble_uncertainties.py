import argparse
import glob, os, re
import pandas as pd
from pathlib import Path
from utils import conf_utils as cu
from utils import logging_utils as lu

if __name__ == "__main__":
    DES5yr = os.getenv("DES5yr")
    DES = os.getenv("DES")

    parser = argparse.ArgumentParser(
        description="Aggregate uncertainties for BNN ensemble methods"
    )

    parser.add_argument(
        "--path_models",
        type=str,
        default=f"{DES}/DES5YR/data_preds/snndump_26XBOOSTEDDES/models/",
        help="Path to models",
    )

    args = parser.parse_args()

    cols_to_keep = ["SNID", "all_class0", "target"]

    for nn in ["variational", "bayesian"]:
        print(nn)
        for my_set in [0, 1, 2]:
            print(my_set)
            list_preds_to_agg = glob.glob(
                f"{args.path_models}/{nn}*1024*/PRED*1024*.pickle"
            )
            list_preds_to_agg = [k for k in list_preds_to_agg if "agg" not in k]
            basename = list_preds_to_agg[0]

            list_df = []
            # now for a given set
            for seed in cu.list_seeds_set[my_set]:
                search_query = re.sub(r"S\_\d+_", f"S_{seed}_", basename)
                try:
                    f = glob.glob(search_query)[0]
                    tmp = pd.read_pickle(f)
                    tmp = tmp[cols_to_keep].copy()
                    list_df.append(tmp)
                except Exception:
                    lu.print_red(f"Seed missing {seed}")

            try:
                df_pred = pd.concat(list_df)

                med_pred = df_pred.groupby("SNID").median()
                med_pred.columns = [str(col) + "_median" for col in med_pred.columns]
                med_pred = med_pred.rename(columns={"target_median": "target"})
                std_pred = df_pred.groupby("SNID").std()
                std_pred.columns = [str(col) + "_std" for col in std_pred.columns]

                del df_pred

                df_bayes = pd.merge(med_pred, std_pred, on=["SNID"])
                df_bayes["SNID"] = df_bayes.index

                outname = re.sub(r"S\_\d+_", f"", basename)
                outname = f"{Path(outname).parent.parent}/{Path(outname).name}"
                outname = outname.replace(
                    ".pickle", f"_aggregated_ensemble_set{my_set}.pickle"
                )

                df_bayes.to_pickle(outname)
                df_bayes.close()
                del df_bayes
            except Exception:
                lu.print_red(f"Missing complete set {my_set}")
