import polars as pl
from polars import col as c
import matplotlib.pyplot as plt
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
    EXPERIMENT_LENGTH,
    IMAGING_RATE,
    CHANNEL1,
    CHANNEL2,
    SLOPE_INDEX,
    SLOPE_MULTIPLIER,
) = gather_input()
time_start = time.time()

for path_to_csv in PATH_TO_CSVS:
    print(f"Proceeding with file: {path_to_csv.stem}")
    # sets up directory for figures, raw find peaks and single cell csvs, outputs pathlib.Paths of each directory
    PATH_TO_FIGURES, PATH_TO_SINGLE_CSVS = setup_directory(
        PATH_TO_SAVING_DIRECTORY, path_to_csv.stem
    )
    full_df = pl.read_csv(path_to_csv)
    cell_IDs = (
        full_df.unique(subset=["Cell_ID"]).select(c("Cell_ID")).to_numpy().flatten()
    )
    total_cells = cell_IDs.size
    i = 0
    for id in cell_IDs:
        i += 1
        print(f"Proceeding with cell {i}/{total_cells}, ID: {id}")
        cell_df = full_df.filter(c("Cell_ID") == id)
        cellgraph = CellGraph(
            id,
            cell_df,
            STARVATION_START,
            STARVATION_END,
            EXPERIMENT_LENGTH,
            IMAGING_RATE,
            CHANNEL1,
            CHANNEL2,
            SLOPE_MULTIPLIER,
            SLOPE_INDEX,
        )

        if (
            cellgraph.birth_frame + 8 <= STARVATION_START < cellgraph.death_frame
            # and id == 26
        ):
            cellgraph.initialize_figure(tick_interval=40)
            cellgraph.graph_base()
            cellgraph.graph_peaks_troughs(PATH_TO_SINGLE_CSVS)
            cellgraph.graph_whi5_exports()
            cellgraph.graph_half_reimport()
            cellgraph.save_figure(PATH_TO_FIGURES)
        else:
            plt.close()

    Final_CSV(PATH_TO_SAVING_DIRECTORY / path_to_csv.stem, PATH_TO_SINGLE_CSVS)
print("End of Analysis")
delta_time = time.time() - time_start
print(f"Finished in: {delta_time}")
