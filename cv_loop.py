import polars as pl
from polars import col as c
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from Array_Pairs_Index_to_Time import pair_peaks
from File_Importer import gather_input, setup_directory
from Final_CSV import Final_CSV
from Graphing import CellGraph
import time

#### INPUT/OUTPUT ####
# opens the UI, gets a bunch of inputs and assigns them to variables. paths are pathlib.Paths,
# STARVATION_START and _end are ints, imaging rate is a float, is_fret is a bool, channels are strings
# TODO: does float64 cause errors?
(
    PATH_TO_CSVS,
    PATH_TO_SAVING_DIRECTORY,
    STARVATION_START,
    STARVATION_END,
    IMAGING_RATE,
    CHANNEL1,
    CHANNEL2,
) = gather_input()
time_start = time.time()

for path_to_csv in PATH_TO_CSVS:
    print(f"Proceeding with file: {path_to_csv.stem}")
    # sets up directory for figures, raw find peaks and single cell csvs, outputs pathlib.Paths of each directory
    PATH_TO_FIGURES, PATH_TO_SINGLE_CSVS = setup_directory(PATH_TO_SAVING_DIRECTORY, path_to_csv.stem)
    full_df = pl.read_csv(path_to_csv)
    cell_IDs = (
        full_df.unique(subset=["Cell_ID"]).select(c("Cell_ID")).to_numpy().flatten()
    )
    #### START LOOPING ####
    for id in cell_IDs:
        # if id > 10: # TZ - cap the for loop at id cells
        #     break
        cell_df = full_df.filter(c("Cell_ID") == id)
        cellgraph = CellGraph(id, cell_df, CHANNEL1, CHANNEL2)

        if cellgraph.frame_of_birth <= STARVATION_START < cellgraph.frame_of_death and cellgraph.lifespan > 5:
            cellgraph.graph_base(STARVATION_START, STARVATION_END)
            cellgraph.graph_starvation_lines(STARVATION_START, STARVATION_END, IMAGING_RATE)
            # TZ - inflection points refer to the points at which 50% of Whi5 is exported from the nucleus
            cellgraph.graph_peaks_troughs(IMAGING_RATE, PATH_TO_SINGLE_CSVS)
            cellgraph.graph_inflection_points(STARVATION_START, IMAGING_RATE)
            cellgraph.graph_slope(IMAGING_RATE, STARVATION_START)
            cellgraph.graph_half_reimport(STARVATION_START)
            cellgraph.save_figure(PATH_TO_FIGURES)
            print(f"Finished with cell {cellgraph.id}")
        else:
            print(f"Cell {id} does not meet conditions")
            plt.close()
    #### STOP LOOPING ####

    # output all csvs in one file
    Final_CSV(PATH_TO_SAVING_DIRECTORY / path_to_csv.stem, PATH_TO_SINGLE_CSVS)
print("End of Analysis")
delta_time = time.time() - time_start
print(f"Finished in: {delta_time}")
