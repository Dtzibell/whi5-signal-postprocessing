import numpy as np
import scipy.signal
def normalize(y: np.ndarray):
    try:
        y_denoised = y - min(y)
        if max(y_denoised) == 0:
            raise ZeroDivisionError
        y_scaled = y_denoised / max(y_denoised)
    except ZeroDivisionError:
        y_scaled = y
    return y_scaled


def smoothen(y, window_length=10, polyorder=3, mode="nearest"):
    y_smooth = scipy.signal.savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode=mode
    )
    return y_smooth

def derive(y, window_length=10, polyorder=3):
    y_derived = scipy.signal.savgol_filter(y, window_length = window_length, polyorder = polyorder, deriv = 1)
    return y_derived
