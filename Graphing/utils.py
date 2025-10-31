import numpy as np
from typing import Iterable
import scipy.signal

def smoothen(y: Iterable, window_length=10, polyorder=3, mode="nearest") -> np.ndarray:
    """
    Smoothens y
    """
    y_smooth: np.ndarray = scipy.signal.savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode=mode
    )
    return y_smooth

def normalize(y: Iterable) -> np.ndarray:
    """
    Denoises the data and scales to 1
    """
    try:
        y_denoised: np.ndarray = y - min(y)
        if max(y_denoised) == 0:
            raise ZeroDivisionError
        y_scaled: np.ndarray = y_denoised / max(y_denoised)
    except ZeroDivisionError:
        y_scaled: np.ndarray = np.array(y)
    return y_scaled

def derive(y: Iterable, window_length=10, polyorder=3) -> np.ndarray:
    """
    Derives y
    """
    y_derived: np.ndarray = scipy.signal.savgol_filter(y, window_length = window_length, polyorder = polyorder, deriv = 1)
    return y_derived
