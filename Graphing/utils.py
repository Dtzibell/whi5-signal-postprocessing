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
    baseline = baseline_fitter.asls(y, lam=1e4, p=0.01)[0]
    return baseline

def get_min(*arrays: np.ndarray) -> float:
    """
    Denoises the data and scales to 1
    """
    arr = np.array([])
    for a in arrays:
        arr = np.append(arr, a)
    _min = np.min(arr)
    return _min

def get_max(*arrays: np.ndarray) -> float:
    """
    Denoises the data and scales to 1
    """
    arr = np.array([])
    for a in arrays:
        arr = np.append(arr, a)
    _max = np.max(arr)
    return _max

def subtract_baseline(x: np.ndarray, y: np.ndarray):
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

def weigh(x: list[float], weights:np.ndarray) -> int:
    """
    Averages array x over array weights.
    @param x: np.ndarray; array which is to be averaged
    @param weights: np.ndarray; weights by which x is to be averaged
    """
    norm_weights = weights / np.sum(weights)
    average_x = np.sum(x * norm_weights)
    return average_x

def normalize(arr: np.ndarray):
    arr_less_min = arr - np.min(arr)
    arr_normalized = arr_less_min / np.max(arr_less_min)
    return arr_normalized
