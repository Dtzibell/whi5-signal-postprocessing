# Min and Max Peaks are numpy arrays, ordered sequentially. Pair the correct max-min values (= Whi5 G1 peak [max]
# with following valley/full nuclear exit [min]) and store in a table. Ignore in between min values that don't correspond to peaks.

import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict

def pair_peaks(path_to_single_csvs, max_peaks, min_peaks, id, imaging_rate):

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
    # print(max_peaks)

    for max_idx,x in enumerate(max_peaks):
        # print(max_idx,x,min_idx)
        try:
            while x > min_peaks[min_idx]:
                min_idx += 1
            else:
                # max_peak_list.append([min_idx,x,id,x*imaging_rate,min_idx*imaging_rate])
                whi5_cycles["Maxima_Index"].append(x)
                whi5_cycles["Minima_Index"].append(min_peaks[min_idx])
                whi5_cycles["Cell_ID"].append(id)
                whi5_cycles["Maxima_Time(min)"].append(x*imaging_rate)
                whi5_cycles["Minima_Time(min)"].append(min_peaks[min_idx]*imaging_rate)
        except IndexError:
            break

    whi5_cycles = pl.DataFrame(whi5_cycles)
    # print(id,whi5_cycles)
    path = path_to_single_csvs / f"Whi5_CV_{id}.csv"
    whi5_cycles.write_csv(path, separator = ",")

    return np.array(whi5_cycles["Minima_Index"])


# If I paste the Array_Pairs body in the for w loop, it seems to be mostly working, though it only logs a few pairs for some cells. Adding the array function
# in the w loop overwrites the csv and gives results only for the last cell ID processed. WHY? 

# When saving individual csv files, it seems it still overwrites the existing one, so if cell ID n has 10 entries, and cell ID n +1 has 9, the final entry of n remains
# in the array. Way to here re-initialise the array when switching cells? >> Put whi5_cycles in the big for loop in Loop_Test. Fixed(?)

# Kinda add the turn indices to time in here in the loop. Add 2 more series next to the index ones(Minima and Maxima) and try to iloc the actual time in min as the loop runs