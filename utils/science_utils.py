import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.295)


def dist_mu(redshift):
    mu = cosmo.distmod(redshift)

    return mu.value


def distance_modulus(df):
    """Add distance modulus
    Args:
        df (DataFrame): with SALT2 fitted features
    Returns:
        df (DataFrame): with distance modulus computed
    """

    # SNIa parameters
    Mb = 19.365
    alpha = 0.144  # from sim
    beta = 3.1
    # Add distance modulus to this Data Frame
    df["mu"] = (
        np.array(df["mB"]) + Mb + np.array(alpha * df["x1"]) - np.array(beta * df["c"])
    )
    df["delmu"] = df["mu"].values - dist_mu(df["zHD"].values.astype(float))
    # assuming theoretical mu nor alpha, beta, abs mag have errors
    df["delmu_err"] = (
        np.array(df["mBERR"])
        + np.array(alpha * df["x1ERR"])
        - np.array(beta * df["cERR"])
    )
    return df


def apply_JLA_cut(df, dump_fits=False):

    if dump_fits:
        # no errors in these sims
        cut_salt_JLA = (df.x1 > -3) & (df.x1 < 3) & (df.c > -0.3) & (df.c < 0.3)
    else:
        cut_salt_JLA = (
            (df.x1 > -3)
            & (df.x1 < 3)
            & (df.c > -0.3)
            & (df.c < 0.3)
            & (df.FITPROB > 0.001)
            & (df.x1ERR < 1)
            & (df.PKMJDERR < 2)
        )
    df_JLA = df[cut_salt_JLA]

    return df_JLA
