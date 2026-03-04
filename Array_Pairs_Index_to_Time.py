# Min and Max Peaks are numpy arrays, ordered sequentially. Pair the correct max-min values (= Whi5 G1 peak [max]
# with following valley/full nuclear exit [min]) and store in a table. Ignore in between min values that don't correspond to peaks.

import numpy as np
import polars as pl
from collections import defaultdict


def pair_peaks(max_peaks, min_peaks, id, imaging_rate, path_to_single_csvs):
    """
    Iterates over max_peaks and min_peaks (simultaneously, always moving forward on the array with the lower iterated
    value and assigns a min_peak to each max_peak. Outputs a np.array of chosen min_peaks and creates a single_cell_csv
    file which has all pairs listed

    :param path_to_single_csvs: pathlib.Path to directory for the file
    :param max_peaks: np.array of max peaks
    :param min_peaks: np.array of min peaks
    :param id: cell id
    :param imaging_rate: imaging rate
    :return: np.array of paired min peaks
    """

    min_idx = 0
    whi5_cycles = defaultdict(list)

    for x in max_peaks:
        try:
            while x > min_peaks[min_idx]:
                min_idx += 1
            whi5_cycles["Cell_ID"].append(id)
            whi5_cycles["Maxima_Index"].append(x)
            whi5_cycles["Minima_Index"].append(min_peaks[min_idx])
            whi5_cycles["Maxima_Time(min)"].append(x * imaging_rate)
            whi5_cycles["Minima_Time(min)"].append(min_peaks[min_idx] * imaging_rate)
        except IndexError:
            break

    whi5_cycles = pl.DataFrame(whi5_cycles)
    path = path_to_single_csvs / f"Whi5_CV_{id}.csv"
    whi5_cycles.write_csv(path, separator=",")

    return np.array(whi5_cycles["Minima_Index"])
