import numpy as np
from typing import Iterable
from numpy.typing import ArrayLike
import polars as pl
import scipy.signal
from pybaselines import Baseline

def smoothen(y: Iterable, window_length=10, polyorder=3, mode="nearest") -> np.ndarray:
    """
    Smoothens y
    """
    y_smooth: np.ndarray = scipy.signal.savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode=mode
    )
    return y_smooth

def get_baseline(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Implements AsLS baseline removal algorithm.

    Read more:
        Baseline Correction with Asymmetric Least Squares Smoothing, Eilers & Boelens, 2005. Contact Tauras for paper if cant find.
    """
    baseline_fitter = Baseline(x_data = x) 
    baseline = baseline_fitter.asls(y, lam=1e6, p=0.01)[0]
    return baseline

def normalize(y: np.ndarray) -> np.ndarray:
    """
    Denoises the data and scales to 1
    """
    temp_y = y-min(y)
    y_scaled: np.ndarray = temp_y / np.max(temp_y)
    return y_scaled

def subtract_baseline(x: pl.Series, y: np.ndarray):
    baseline = get_baseline(x, y)
    y_growth = np.maximum(y - baseline, 0)
    # y_growth = temp_y / np.max(temp_y)
    return y_growth

def derive(y: ArrayLike, degree) -> np.ndarray:
    """
    Derives y
    """
    y_derived: np.ndarray = np.gradient(y)
    for _ in range(degree-1):
        y_derived: np.ndarray = np.gradient(y_derived)
    return y_derived
