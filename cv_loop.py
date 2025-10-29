import polars as pl
from polars import col as c
import numpy as np
import matplotlib.pyplot as plt
from polars.exceptions import ColumnNotFoundError  # .exceptions instead of .polars?
from scipy import signal
from scipy.signal import find_peaks
from Array_Pairs_Index_to_Time import pair_peaks
from File_Importer import gather_input, setup_directory
from Final_CSV import Final_CSV
from TZfunctions import (
    extract_x_and_y,
    normalize_to_1,
    find_inflection_points,
    filter_inflection_points,
    find_change_point,
    find_high_point,
    prepare_df,
    extract_x_and_y
)
import time

#### INPUT/OUTPUT ####
# opens the UI, gets a bunch of inputs and assigns them to variables. paths are pathlib.Paths,
# STARVATION_START and _end are ints, imaging rate is a float, is_fret is a bool, channels are strings
# TODO: does float64 cause errors?
(
    PATH_TO_FILES,
    PATH_TO_DIRECTORY,
    STARVATION_START,
    STARVATION_END,
    IMAGING_RATE,
    CHANNEL_1,
    CHANNEL_2,
) = gather_input()
time_start = time.time()
MINUTES_STARVATION_START = STARVATION_START * IMAGING_RATE
MINUTES_STARVATION_END = STARVATION_START * IMAGING_RATE

for path in PATH_TO_FILES:
    print(f"Proceeding with file: {path.stem}")
    # sets up directory for figures, raw find peaks and single cell csvs, outputs pathlib.Paths of each directory
    figures_path, single_csvs_path = setup_directory(PATH_TO_DIRECTORY, path.stem)
    full_df, unique_IDs = prepare_df(path)
    #### INPUT/OUTPUT ####

    #### START LOOPING ####
    for id in unique_IDs:
        # if id > 10: # TZ - cap the for loop at id cells
        #     break
        cell_df = full_df.filter(c("Cell_ID") == id)
        x_time, y_signal = extract_x_and_y(cell_df, CHANNEL_1, CHANNEL_2)

        if min(x_time) <= MINUTES_STARVATION_START < max(x_time) and len(y_signal) > 5:
            # in words: does not go any further if cell is born after start of starvation,
            # disappears before start of starvation or is only on the screen for 5 frames

            # AP - After filtering, smooth and continue
            y_signal_smooth = signal.savgol_filter(
                y_signal, window_length=10, polyorder=3, mode="nearest"
            )
            y_signal_scaled = normalize_to_1(y_signal_smooth)


            # find peaks
            # TZ - distance was originally 25, but had to be reduced because peaks were being detected too sparsely
            # can increase again if some beautiful plots are wanted, but for functionality it's bad
            peaks = np.array(find_peaks(y_signal_smooth, distance=10)[0])
            troughs = np.array(find_peaks(- y_signal_smooth)[0])
            peaks_minutes = peaks * IMAGING_RATE
            troughs_minutes = troughs * IMAGING_RATE

            # TZ - each cell has an individual starvation onset which is dependent on its birth time
            individual_starvation_onset = MINUTES_STARVATION_START - min(x_time)
            individual_end_of_starvation = MINUTES_STARVATION_END - min(x_time)

            # TZ - inflection points refer to the points at which 50% of Whi5 is exported from the nucleus
            inflection_points = find_inflection_points(y_signal_scaled)

            paired_troughs = pair_peaks(
                peaks, troughs, id, IMAGING_RATE, single_csvs_path
            )
            filtered_inflection_points, difference_from_starvation_onset = (
                filter_inflection_points(
                    peaks,
                    paired_troughs,
                    inflection_points,
                    individual_starvation_onset,
                )
            )


            # TZ - find values in starvation range
            y_signal_starvation = [
                y
                for i, y in enumerate(y_signal_scaled)
                if individual_starvation_onset
                <= i * IMAGING_RATE
                <= individual_end_of_starvation
            ]
            first_deriv_smooth = signal.savgol_filter(
                y_signal_scaled, window_length=10, polyorder=3, deriv=1
            )
            first_deriv_smooth_starvation = [
                s
                for i, s in enumerate(first_deriv_smooth)
                if individual_starvation_onset
                <= i * IMAGING_RATE
                <= individual_end_of_starvation
            ]
            y_signal_starvation, first_deriv_smooth_starvation = (
                np.array(y_signal_starvation),
                np.array(first_deriv_smooth_starvation),
            )
            first_deriv_smooth_starvation_scaled = first_deriv_smooth_starvation / max(
                first_deriv_smooth_starvation
            )

            # TZ - find low and high points
            point_low_x, point_low_y = find_change_point(
                y_signal_starvation, first_deriv_smooth_starvation_scaled, 4, 0.005
            )
            point_high_x, point_high_y = find_high_point(y_signal_starvation)

            # TZ - tester prints
            # print(f"low_x: {point_low_x}")
            # print(f"low_y: {point_low_y}")
            # print(f"high_x: {point_high_x}")
            # print(f"high_y: {point_high_y}")

            if (
                point_high_x is not None
                and point_low_x is not None
                and point_low_x < point_high_x
            ):
                # TZ - correct for time at birth
                point_low_x = int(
                    round(
                        (point_low_x * IMAGING_RATE + individual_starvation_onset)
                        / IMAGING_RATE
                    )
                )
                point_high_x = int(
                    round(
                        (point_high_x * IMAGING_RATE + individual_starvation_onset)
                        / IMAGING_RATE
                    )
                )
                # print(type(point_low_x))

                # TZ - find slope of the line connecting the chosen points
                x_values_points_RiseOfSlope = [
                    point_low_x * IMAGING_RATE + min(x_time),
                    point_high_x * IMAGING_RATE + min(x_time),
                ]
                y_values_points_RiseOfSlope = [point_low_y, point_high_y]
                Slope_RiseOfSlope = np.gradient(
                    y_values_points_RiseOfSlope, x_values_points_RiseOfSlope
                )

                # TZ - tester prints
                # print(f"Slope: {Slope_RiseOfSlope}")
                # print(f"y_values: {y_values_points_RiseOfSlope}")
                # print(f"low point x,y: {point_low_x, point_low_y}")
                # print(f"y_starvation: {y_signal_starvation}")
                # print(f"x values: {x_values_points_RiseOfSlope}")
                # print(f"x_time:{list(x_time)}")

                # TZ - TODO: can implement selection of points through growth of values. Sometimes works better,\
                #       because slope growth model is quite bad at recognizing instant reimports.

            else:
                # TZ - if no points are found, assign value to toggle off slope plotting for the cell
                point_low_x = None
                point_high_x = None

            f, ax = plt.subplots(figsize=(8, 8))

            # Axes Names and Plot Title
            plt.xlabel("Time (mins)")
            plt.ylabel("Norm. Whi5 CV from Full Cell Mask")
            plt.title("Normalized Whi5 CV over Time")

            # GRAPH
            plt.plot(
                x_time, y_signal_scaled, color="k"
            )  # Switched to black signal line
            plt.xlim(0, max(x_time))

            # VLINES
            # if is_starvation == "Yes":
            plt.vlines(x=MINUTES_STARVATION_START, ymin=0, ymax=max(y_signal_scaled), color="r")
            plt.vlines(x=MINUTES_STARVATION_END, ymin=0, ymax=max(y_signal_scaled), color="r")

            if filtered_inflection_points is not None:
                plt.vlines(
                    x=[
                        i * IMAGING_RATE + min(x_time)
                        for i in filtered_inflection_points
                    ],
                    ymin=0,
                    ymax=max(y_signal_scaled),
                    color="b",
                )
            # this has to be iterated over because np.arrays cannot be multiplied by floats.

            # PEAKS
            if paired_troughs is not None:
                plt.plot(x_time[peaks], y_signal_scaled[peaks], "*")
                plt.plot(
                    x_time[paired_troughs], y_signal_scaled[paired_troughs], "v"
                )

            # SLOPES
            if (
                point_high_x is not None
                and point_low_x is not None
                and point_low_x < point_high_x
            ):
                # TZ - plot points, line between them and the slope of the connecting line
                plt.plot(
                    x_time[point_low_x],
                    y_signal_scaled[point_low_x],
                    ".",
                    color="g",
                    markersize=10,
                )
                plt.plot(
                    x_time[point_high_x],
                    y_signal_scaled[point_high_x],
                    ".",
                    color="g",
                    markersize=10,
                )
                plt.plot(
                    x_values_points_RiseOfSlope,
                    y_values_points_RiseOfSlope,
                    color="g",
                    linestyle="--",
                )
                plt.text(
                    point_high_x * IMAGING_RATE - 50 + min(x_time),
                    point_high_y + 0.05,
                    f"Slope {round(Slope_RiseOfSlope[0] * 100, 4)}",
                )  # TZ - labels the top point with AU (slope*100)
                plt.text(
                    point_low_x * IMAGING_RATE + 12 + min(x_time),
                    point_low_y,
                    point_low_x * IMAGING_RATE + min(x_time),
                )
            else:
                pass

            # SCALE
            plt.xticks(np.arange(0, max(x_time), 40), rotation=90)  # TZ - adjust X axis

            # TEXT
            try:
                for idx, p in enumerate(
                    filtered_inflection_points
                ):  # TZ - plot for all items in infls_filtered2
                    # TZ - plots the difference between starvation onset and filtered inflection points at the top of the vertical lines that the inflection points are plotted as (in minutes)
                    plt.text(
                        p * IMAGING_RATE + 5 + min(x_time),
                        0.95,
                        str(difference_from_starvation_onset[idx] * IMAGING_RATE),
                        rotation=90,
                    )
            except TypeError:  # in case filtered_inflection_points is None
                pass

            # Save figures
            fig = plt.gcf()
            fig.savefig(
                figures_path / f"Cell_{id}_Whi5_CV.png", bbox_inches="tight", dpi=300
            )
            plt.close()
            print(f"Finished with cell {id}")
        else:
            print(f"Cell {id} does not meet conditions")
    #### STOP LOOPING ####

    # output all csvs in one file
    Final_CSV(concat_path_name, single_csvs_path)
print("End of Analysis")
delta_time = time.time() - time_start
print(f"Finished in: {delta_time}")
