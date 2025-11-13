# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:54:45 2025

@author: 37067
"""
# DISCLAIMER: These are not meant to be universal functions suitable for any code.
# Implementing in other codes may require significant changes.

import pathlib
from typing import Iterable
import numpy as np
from scipy import ndimage as ndim
import polars as pl
import matplotlib.pyplot as plt
import scipy.signal
from Graphing.utils import (
    derive,
    normalize,
    smoothen,
)
from collections import defaultdict


class CellGraph:
    def __init__(
        self, id: int, cell_df: pl.DataFrame, CHANNEL1: str, CHANNEL2: str
    ) -> None:
        """
        Initializes the CellGraph object.
        @param id: int; Cell ID.
        @param cell_df: pl.DataFrame; Data of cell
        @param CHANNEL1: str; name of first channel that is used for data extraction
        @param CHANNEL2: str; name of second channel that is used for data extraction. Applies for FRET experiments.
        Adds the following attributes to the object:
            self.cell_df
            self.id
            self.x: pl.Series; temporally sorted values of x in minutes
            self.y: pl.Series; temporally sorted values of y
            self.birth_frame: int; frame of cell's appearance
            self.death_frame: int; frame of cell's disappearance
            self.lifespan: int; frames within image
        """
        self.cell_df: pl.DataFrame = cell_df
        self.id: int = id
        self.x: pl.Series = self.extract_x()
        self.y: pl.Series = self.extract_y(CHANNEL1, CHANNEL2)
        self.birth_frame: int = round(min(self.x) / 3)
        self.death_frame: int = round(max(self.x) / 3)
        self.lifespan: int = len(self.y)

    def initialize_figure(self, IMAGING_RATE: float, tick_interval: float = 40) -> None:
        """
        Initializes and captions the plot
        @param IMAGING_RATE: float; imaging rate in minutes
        @param tick_interval: float; time between two x ticks of the figure
        Adds the following attributes to the object:
            self.figure: matplotlib.figure.Figure; more under https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
            self.axes: matplotlib.axes.Axes; more under https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
        """
        self.figure, self.axes = plt.subplots(figsize=(8, 8))
        plt.xlabel("Time (mins)")
        plt.ylabel("Norm. Whi5 CV from Full Cell Mask")
        plt.title("Normalized Whi5 CV over Time")
        plt.xticks(
            np.arange(0, self.death_frame * IMAGING_RATE, tick_interval), rotation=90
        )

    def graph_base(self) -> None:
        """
        Smoothens and scales CellGraph.y
        Sets plot's x and y axis limits and plots CellGraph.y over CellGraph.x
        Adds the following attributes to the object:
            self.y_scaled: np.ndarray; y that is smoothened and scaled to 1.
        """
        y_smooth: np.ndarray = smoothen(self.y)
        self.y_scaled: np.ndarray = normalize(y_smooth)
        # TODO: would make sense to have a "length_of_experiment" field to cap x, not individual death_frame
        plt.xlim(0, max(self.x))
        plt.ylim(-0.1, 1.1)
        plt.plot(self.x, self.y_scaled, color="k")

    def extract_x(self) -> pl.Series:
        """
        Retrieves time in minutes from CellGraph.cell_df
        """
        x: pl.Series = self.cell_df["time_minutes"]
        return x

    def extract_y(self, CHANNEL1: str, CHANNEL2: str) -> pl.Series:
        """
        Retrieves the signal of interest from CellGraph.cell_df
        @param CHANNEL1: str; name of first channel that is used for data extraction
        @param CHANNEL2: str; name of second channel that is used for data extraction. Applies for FRET experiments.
        """
        if CHANNEL2 is not "":
            y_signal_1: pl.Series = self.cell_df[CHANNEL1]
            y_signal_2: pl.Series = self.cell_df[CHANNEL2]
            y: pl.Series = y_signal_1 / y_signal_2
        else:
            y: pl.Series = self.cell_df[CHANNEL1]  # e.g. Quad2_mCherry_CV
        return y

    def graph_starvation_lines(
        self, STARVATION_START: int, STARVATION_END: int, IMAGING_RATE: float
    ) -> None:
        """
        Plots vertical lines indicating starvation start and end.
        @param STARVATION_START: int; frame of start of starvation
        @param STARVATION_END: int; frame of end of starvation
        @param IMAGING_RATE: float; imaging rate in minutes
        Adds the following attributes to the object:
            self.starvation_start: int; cell birth frame adjust starvation start
            self.starvation_end: int; cell death frame adjust starvation end
        """
        self.starvation_start: int = STARVATION_START - self.birth_frame
        self.starvation_end: int = STARVATION_END - self.death_frame
        plt.vlines(x=STARVATION_START * IMAGING_RATE, ymin=0, ymax=1)
        plt.vlines(x=STARVATION_END * IMAGING_RATE, ymin=0, ymax=1)

    def graph_peaks_troughs(
        self, IMAGING_RATE: float, SINGLE_CSV_SAVING_DIR: pathlib.Path
    ) -> None:
        """
        Saves whi5 cycles in a csv and plots their peaks and troughs.
        @param IMAGING_RATE: float; imaging rate in minutes
        @param SINGLE_CSV_SAVING_DIR: pathlib.Path; path where single cell whi5 cycles are saved.
        Adds the following attributes to the object:
            self.peaks: np.ndarray; peaks of CellGraph.y_scaled
            self.troughs: np.ndarray; troughs of CellGraph.y_scaled
            self.whi5_cycles: pl.DataFrame; Data of Whi5 cycling
            self.paired_troughs: np.ndarray; Troughs that are preceded by a peak
        """
        self.peaks: np.ndarray = self.get_peaks()
        self.troughs: np.ndarray = self.get_troughs()
        self.whi5_cycles: pl.DataFrame = self.save_whi5_cycles(
            IMAGING_RATE, SINGLE_CSV_SAVING_DIR
        )
        self.paired_troughs: pl.Series = self.get_paired_troughs()
        plt.plot(self.x[self.peaks], self.y_scaled[self.peaks], "^")
        plt.plot(self.x[self.paired_troughs], self.y_scaled[self.paired_troughs], "v")

    def get_peaks(self, distance = None) -> np.ndarray:
        """
        Finds CellGraph.y_scaled peaks.
        @param distance: int; smallest amount of frames between two peaks
        """
        peaks: np.ndarray = np.array(
            scipy.signal.find_peaks(self.y_scaled, distance=distance)[0]
        )
        proms = scipy.signal.peak_prominences(self.y_scaled, peaks)[0]
        contour_heights = self.y_scaled[peaks] - proms
        # print(len(self.y_scaled), proms)
        plt.vlines(x=peaks*3, ymax = self.y_scaled[peaks], ymin = contour_heights)
        return peaks

    def get_troughs(self) -> np.ndarray:
        """
        Finds CellGraph.y_scaled troughs
        """
        troughs: np.ndarray = np.array(scipy.signal.find_peaks(-self.y_scaled)[0])
        proms = scipy.signal.peak_prominences(-self.y_scaled, troughs)[0]
        contour_heights = self.y_scaled[troughs] + proms
        plt.vlines(x=troughs*3, ymin = self.y_scaled[troughs], ymax = contour_heights)
        return troughs

    def save_whi5_cycles(
        self, IMAGING_RATE: float, SINGLE_CSV_SAVING_DIR: pathlib.Path
    ) -> pl.DataFrame:
        """
        Iterates over peaks and finds the first trough that comes right after each of them.
        @param IMAGING_RATE: float; imaging rate in minutes
        @param SINGLE_CSV_SAVING_DIR: pathlib.Path; path where single cell whi5 cycles are saved.
        """
        min_idx: int = 0
        cycler: defaultdict = defaultdict(list)

        for x in self.peaks:
            try:
                while x > self.troughs[min_idx]:
                    min_idx += 1
                cycler["Maxima_Index"].append(x)
                cycler["Minima_Index"].append(self.troughs[min_idx])
                cycler["Cell_ID"].append(self.id)
                cycler["Maxima_Time(min)"].append(x * IMAGING_RATE)
                cycler["Minima_Time(min)"].append(self.troughs[min_idx] * IMAGING_RATE)
            except IndexError:
                break

        whi5_cycles: pl.DataFrame = pl.DataFrame(cycler)
        path: pathlib.Path = SINGLE_CSV_SAVING_DIR / f"Whi5_CV_{id}.csv"
        whi5_cycles.write_csv(path, separator=",")

        return whi5_cycles

    def get_paired_troughs(self) -> pl.Series:
        """
        Gets troughs that are preceded by a peak
        """
        return self.whi5_cycles["Minima_Index"]

    def graph_whi5_exports(self, IMAGING_RATE):
        """
        Finds whi5 exports and graphs them.
        @param IMAGING_RATE: int; imaging rate in minutes
        Adds the following attributes to the object:
            self.exports_of_interest: list[int]; whi5 exports that lay with time_of_interest from STARVATION_START in minutes
            self.times_to_starvation: list[float]; time until STARVATION_START in minutes 
        """
        raw_inflection_points: np.ndarray = self.find_inflection_points()
        whi5_exports: list[int] = self.filter_whi5_exports(raw_inflection_points)
        self.exports_of_interest, self.times_to_starvation = (
            self.filter_within_time(whi5_exports, 120, IMAGING_RATE)
        )
        plt.vlines(
            x=[i for i in self.exports_of_interest],
            ymin=0,
            ymax=1,
            color="b",
        )
        for idx in range(len(self.exports_of_interest)):
            plt.text(
                self.exports_of_interest[idx] + 5,
                0.95,
                str(self.times_to_starvation[idx]),
                rotation=90,
            )

    def find_inflection_points(self) -> np.ndarray:
        """
        Gets inflection points
        """
        second_derivative: np.ndarray = ndim.gaussian_filter1d(
            np.gradient(np.gradient(self.y_scaled)), 2
        )
        inflection_points: np.ndarray = np.where(np.diff(np.sign(second_derivative)))[0]
        return inflection_points

    def filter_whi5_exports(self, iter: Iterable) -> list[int]:
        """
        filters an iterable for items that are between CellGraph's peak and trough.
        @param iter: Iterable;
        """
        items_between_peak_and_trough: list[int] = []
        for i in range(len(self.peaks) - 1):
            for item in iter:
                is_between_peak_and_trough: bool = (
                    self.peaks[i] < item < self.paired_troughs[i]
                )
                if is_between_peak_and_trough:
                    items_between_peak_and_trough.append(item)

        return items_between_peak_and_trough

    def filter_within_time(
        self, iter: Iterable, time_of_interest: int, IMAGING_RATE: float
    ):
        """
        Filter an iterable for items that lay within time_of_interest to STARVATION_START
        @param iter: Iterable;
        @param time_of_interest: int;
        @param IMAGING_RATE: float, imaging rate in minutes
        """
        frames_of_interest: int = round(time_of_interest / IMAGING_RATE)
        inflection_points_of_interest: list[float] = []
        times_to_starvation: list[float] = []

        for p in iter:
            frames_to_starvation = self.starvation_start - p
            if 0 < frames_to_starvation < frames_of_interest:
                inflection_points_of_interest.append(
                    (self.birth_frame + p) * IMAGING_RATE
                )
                times_to_starvation.append((self.starvation_start - p) * IMAGING_RATE)

        return inflection_points_of_interest, times_to_starvation

    def graph_slope(self, IMAGING_RATE, STARVATION_START):
        """
        Retrieves the reimport onset, the end of reimport and the slope between those two points and plots them.
        @param IMAGING_RATE: float, imaging rate in minutes
        @param STARVATION_START: int; frame of start of starvation
        Adds the following attributes to the object:
            self.slopes: np.ndarray; the first derivative of CellGraph.y_scaled
            self.y_starvation: np.ndarray; scaled y within starvation
            self.slopes_starvation: np.array;
            self.reimport_onset_x: int;
            self.reimport_onset_y: int;
            self.end_of_reimport_x: int;
            self.end_of_reimport_y: int;
            self.slope_of_slope: float; the slope of the slope between reimport onset and end of reimport
            self.x_of_slope: list[int]; x coordinates of the slope
            self.y_of_slope: list[float]; y coordinates of the slope

        """
        self.slopes: np.ndarray = derive(self.y_scaled)
        self.y_starvation, self.slopes_starvation = self.filter_for_starvation(
            self.y_scaled, self.slopes
        )
        self.reimport_onset_x, self.reimport_onset_y = self.find_reimport_onset(
            STARVATION_START
        )
        self.end_of_reimport_x, self.end_of_reimport_y = self.find_end_of_reimport(STARVATION_START)
        if (
            self.reimport_onset_y != -1
            and self.end_of_reimport_y != -1
            and self.reimport_onset_x < self.end_of_reimport_x
        ):
            self.slope_of_slope, self.x_of_slope, self.y_of_slope = (
                self.get_slope_of_reimport(IMAGING_RATE)
            )
            plt.plot(
                self.reimport_onset_x * IMAGING_RATE,
                self.reimport_onset_y,
                ".",
                color="g",
                markersize=10,
            )
            plt.plot(
                self.end_of_reimport_x * IMAGING_RATE,
                self.end_of_reimport_y,
                ".",
                color="g",
                markersize=10,
            )
            plt.plot(self.x_of_slope, self.y_of_slope, color="g", markersize=10)
            plt.text(
                self.end_of_reimport_x * IMAGING_RATE - 50,
                self.end_of_reimport_y + 0.05,
                f"Slope: {round(self.slope_of_slope[0] * 100, 4)}",
            )
            plt.text(
                self.reimport_onset_x * IMAGING_RATE + 12,
                self.reimport_onset_y,
                self.reimport_onset_x * IMAGING_RATE,
            )

    def filter_for_starvation(self, *args: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Iterates over *args and selects their indices that fall within starvation
        @param *args: np.ndarray;
        """
        filtered: list[np.ndarray] = []
        for i in args:
            filtered.append(i[self.starvation_start : self.starvation_end])
        filtered_tuple: tuple[np.ndarray, ...] = tuple(filtered)
        return filtered_tuple

    def find_reimport_onset(self, STARVATION_START: int, window: int = 4, factor: float = 0.0005) -> tuple[int, float]:
        """
        Finds the frame and the signal value where the cell starts whi5 reimport.
        @param STARVATION_START: int; frame of start of starvation
        @param window: int; amount of points to consider. Higher values are more sensitive to slow reimport.
        @param factor: float; the amount that a series of slopes must have increased for point 0 to be recognized as reimport onset. The larger it is, the more the slopes have to increase with each step.
        """
        for idx, y in enumerate(self.y_starvation):
            slopes_in_window = self.slopes_starvation[idx : idx + window]
            if (slopes_in_window[-1] / slopes_in_window[0]) > (
                (1 + factor / slopes_in_window[0]) ** window
            ) and all(s > 0 for s in slopes_in_window):
                reimport_onset_x = idx + STARVATION_START
                reimport_onset_y = y
                return reimport_onset_x, reimport_onset_y
        else:
            return -1, -1

    def find_end_of_reimport(self, STARVATION_START: int, factor: float = 0.8) -> tuple[int, float]:
        """
        Finds the first point that is larger than the maximum y in starvation multiplied by a factor
        @param STARVATION_START: int; frame of start of starvation
        @param factor: float; the number that the maximum y is multiplied by. The higher it is, the higher the point will be set.
        """
        for idx, y in enumerate(self.y_starvation):
            if y > factor * (max(self.y_starvation)):
                end_of_reimport_x: int = idx + STARVATION_START
                end_of_reimport_y: float = y
                return end_of_reimport_x, end_of_reimport_y
        else:
            return -1, -1

    def get_slope_of_reimport(self, IMAGING_RATE) -> tuple[np.ndarray, list[float], list[float]]:
        """
        Finds the slope of whi5 reimport.
        @param IMAGING_RATE: float, imaging rate in minutes
        """
        x_of_slope: list[float] = [
            self.reimport_onset_x * IMAGING_RATE,
            self.end_of_reimport_x * IMAGING_RATE,
        ]
        y_of_slope: list[float] = [self.reimport_onset_y, self.end_of_reimport_y]
        slope_of_slope: np.ndarray = np.gradient(y_of_slope, x_of_slope)
        return slope_of_slope, x_of_slope, y_of_slope

    def graph_half_reimport(self, STARVATION_START):
        try:
            last_inflection_point = round(self.exports_of_interest[-1] / 3)
            print(last_inflection_point)
            value_at_inflection_point = self.y_scaled[last_inflection_point - self.birth_frame]
            print(value_at_inflection_point)
            for idx, val in enumerate(self.y_starvation):
                if val > value_at_inflection_point:
                    first_encounter_within_starvation = (
                        (idx + STARVATION_START) * 3,
                        val,
                    )
                    plt.plot(
                        first_encounter_within_starvation[0],
                        first_encounter_within_starvation[1],
                        ".",
                        markersize=20,
                        color="g",
                    )
                    plt.text(
                        first_encounter_within_starvation[0],
                        first_encounter_within_starvation[1] - 0.05,
                        f"val = {first_encounter_within_starvation[0]}",
                    )
                    break
        except IndexError:  # no inflection points present
            pass

    def get_individual_starvation_start(self, STARVATION_START: int):
        """
        Gets cell's individual starvation start
        @param STARVATION_START: int; frame of start of starvation
        """
        starvation_start = STARVATION_START - self.birth_frame
        return starvation_start

    def get_individual_starvation_end(self, STARVATION_END: int):
        """
        Gets cell's individual starvation end
        @param STARVATION_START: int; frame of start of starvation
        """
        starvation_end = STARVATION_END - self.death_frame
        return starvation_end

    def save_figure(self, PATH_TO_FIGURES: pathlib.Path):
        """
        Saves the figure.
        @param PATH_TO_FIGURES: pathlib.Path;
        """
        fig = plt.gcf()
        fig.savefig(
            PATH_TO_FIGURES / f"Cell_{self.id}_Whi5_CV.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
