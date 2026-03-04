from numpy import full
import polars as pl
from polars import col as c
import matplotlib.pyplot as plt
from polars import schema
from File_Importer import gather_input, setup_directory
from Final_CSV import Final_CSV
from Graphing import CellGraph
import time
from configparser import ConfigParser
from pathlib import Path

from Graphing.cell_graph import Lifespan

config = ConfigParser()
config.read("config.ini")
PATH_TO_OUTPUT_DIR = Path(config["PATHS"]["ResultsDirectory"])
print(f"The output will be saved to {PATH_TO_OUTPUT_DIR}")

(
    PATH_TO_CSVS,
    STARVATION_START,
    STARVATION_END,
    CHANNEL1,
    CHANNEL2,
    SLOPE_INDEX,
    SLOPE_MULTIPLIER,
) = gather_input()
time_start = time.time()

for csv in PATH_TO_CSVS:
    print(f"Proceeding with file: {csv.stem}")
    # sets up directory for figures, raw find peaks and single cell csvs, outputs pathlib.Paths of each directory
    PATH_TO_FIGURES, PATH_TO_SINGLE_CSVS = setup_directory(PATH_TO_OUTPUT_DIR, csv.stem)
    relevant_columns = [
        "time_minutes",
        "frame_i",
        "Cell_ID",
        CHANNEL1,
    ]
    if CHANNEL2 != "":
        relevant_columns.append(CHANNEL2)
    full_df = pl.scan_csv(csv).select(relevant_columns).collect()
    partitioned_df = full_df.partition_by("Cell_ID", as_dict=True)
    EXPERIMENT_LENGTH = full_df[-1, "frame_i"]
    IMAGING_RATE = (
        partitioned_df[(1,)][1, "time_minutes"]
        - partitioned_df[(1,)][0, "time_minutes"]
    )
    cell_IDs = full_df.get_column("Cell_ID").unique()
    # [0] gets the height of the series
    total_cells = cell_IDs.shape[0]
    i = 0
    for key in partitioned_df.keys():
        i += 1
        id = key[0]
        print(f"Proceeding with cell {i}/{total_cells}, ID: {int(id)}")
        cell_df = partitioned_df[key]
        cellgraph = CellGraph(
            round(int(id)),
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
        if cellgraph.cell_df.height < cellgraph.lifespan.end - cellgraph.lifespan.start:
            continue

        if (
            cellgraph.get_birth_frame() + 8
            <= STARVATION_START
            < cellgraph.get_death_frame()
            # and id == 26
        ):
            cellgraph.initialize_figure()
            cellgraph.graph_full()
            cellgraph.graph_prestarvation()
            if cellgraph.has_starvation:
                cellgraph.graph_starvation()
            if cellgraph.has_poststarvation:
                cellgraph.graph_poststarvation()
            cellgraph.graph_peaks_troughs(PATH_TO_SINGLE_CSVS)
            cellgraph.graph_whi5_exports()
            cellgraph.graph_half_reimport()
            cellgraph.format_figure()
            cellgraph.save_figure(PATH_TO_FIGURES)
        else:
            plt.close()

    Final_CSV(PATH_TO_OUTPUT_DIR / csv.stem, PATH_TO_SINGLE_CSVS)
print("End of Analysis")
delta_time = time.time() - time_start
print(f"Finished in: {delta_time}")
