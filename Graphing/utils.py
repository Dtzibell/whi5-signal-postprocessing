import numpy as np
from typing import Iterable
from numpy.typing import ArrayLike
import scipy.signal
from pybaselines import Baseline
import polars as pl
from polars import col as c


def smoothen(
    df: pl.DataFrame,
    y_name: str = "signal",
    window_length=10,
    polyorder=3,
    mode="nearest",
) -> pl.DataFrame:
    """
    Smoothens y
    """
    y = df[y_name].to_numpy()
    signal_smooth = scipy.signal.savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode=mode
    )
    df = df.with_columns(pl.Series("signal_smooth", signal_smooth))
    return df

    y_smooth: np.ndarray = scipy.signal.savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode=mode
    )
    return y_smooth


def get_baseline(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Implements AsLS baseline removal algorithm.

    Read more:
        Baseline Correction with Asymmetric Least Squares Smoothing,
        Eilers & Boelens, 2005.
    """
    baseline_fitter = Baseline(x_data=x)
    baseline = baseline_fitter.asls(y, lam=1e4, p=0.01)[0]
    return baseline


def subtract_baseline(
    df: pl.DataFrame, x_name: str, y_name: str = "signal"
) -> pl.DataFrame:
    x, y = df[x_name], df[y_name]
    baseline = get_baseline(x, y)
    y_growth = np.maximum(y - baseline, 0)
    df = df.with_columns(pl.Series("signal_less_bl", y_growth))
    return df


def derive(df: pl.DataFrame, degree: int, y_name: str = "signal") -> pl.DataFrame:
    """
    Derives y
    """
    y = df[y_name]
    y_derived: np.ndarray = np.gradient(y)
    for _ in range(degree - 1):
        y_derived: np.ndarray = np.gradient(y_derived)
    df = df.with_columns(pl.Series("signal_d1y", y_derived))
    return df


def weigh(x: list[float] | np.ndarray, weights: np.ndarray) -> int:
    """
    Averages array x over array weights.
    @param x: np.ndarray; array which is to be averaged
    @param weights: np.ndarray; weights by which x is to be averaged
    """
    norm_weights = weights / np.sum(weights)
    average_x = np.sum(x * norm_weights)
    return average_x


def normalize(df: pl.DataFrame, y_name: str = "signal") -> pl.DataFrame:
    y: np.ndarray = df[y_name].to_numpy()
    y_less_min = y - np.min(y)
    y_normalized = y_less_min / np.max(y_less_min)
    df = df.with_columns(pl.Series("signal_normalized", y_normalized))
    return df
