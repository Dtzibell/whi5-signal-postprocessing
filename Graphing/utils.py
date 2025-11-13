import numpy as np
from typing import Iterable
from numpy.typing import ArrayLike
import polars
import scipy.signal
from pybaselines import Baseline
import matplotlib.pyplot as plt

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
    y_scaled: np.ndarray = (y - min(y)) / np.max(y)
    return y_scaled

def subtract_baseline(x: polars.Series, y: np.ndarray):
    baseline = get_baseline(x, y)
    temp_y = np.maximum(y - baseline, 0)
    y_growth = temp_y / np.max(temp_y)
    return y_growth

def derive(y: Iterable, window_length=4, polyorder=2) -> np.ndarray:
    """
    Derives y
    """
    y_derived: np.ndarray = scipy.signal.savgol_filter(y, window_length = window_length, polyorder = polyorder, deriv = 1)
    return y_derived
