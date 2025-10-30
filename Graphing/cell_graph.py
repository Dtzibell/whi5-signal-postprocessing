# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:54:45 2025

@author: 37067
"""
# DISCLAIMER: These are not meant to be universal functions suitable for any code.
# Implementing in other codes may require significant changes.

import pathlib
import numpy as np
from scipy import ndimage as ndim
import polars as pl
from polars import col as c
import matplotlib.pyplot as plt
import scipy.signal
from Graphing.utils import (
    derive,
    normalize,
    smoothen,
)
from collections import defaultdict


class CellGraph:
    def __init__(self, id: int, cell_df: pl.DataFrame, CHANNEL1, CHANNEL2):
        self.cell_df: pl.DataFrame = cell_df
        self.id: int = id

        self.figure, self.axes = plt.subplots(figsize=(8, 8))
        plt.xlabel("Time (mins)")
        plt.ylabel("Norm. Whi5 CV from Full Cell Mask")
        plt.title("Normalized Whi5 CV over Time")

        self.x: pl.Series = self.extract_x()
        self.frame_of_birth: int = round(min(self.x) / 3)
        self.frame_of_death: int = round(max(self.x) / 3)
        self.y: pl.Series = self.extract_y(CHANNEL1, CHANNEL2)
        self.lifespan: int = len(self.y)

    def graph_base(self, STARVATION_START, STARVATION_END):
        self.y_smooth: np.ndarray = smoothen(self.y)
        self.y_scaled: np.ndarray = normalize(self.y_smooth)
        self.get_individual_starvation_timing(STARVATION_START, STARVATION_END)

        plt.xlim(0, max(self.x))
        plt.ylim(-0.1, 1.1)
        plt.plot(self.x, self.y_scaled, color="k")

    def extract_x(self):
        x: pl.Series = self.cell_df["time_minutes"]
        return x

    def extract_y(self, CHANNEL1: str, CHANNEL2: str):
        if CHANNEL2 is not "":
            y_signal_1 = self.cell_df[CHANNEL1]
            y_signal_2 = self.cell_df[CHANNEL2]
            y: pl.Series = y_signal_1 / y_signal_2
        else:
            y: pl.Series = self.cell_df[CHANNEL1]  # e.g. Quad2_mCherry_CV
        return y

    def graph_starvation_lines(
        self, STARVATION_START: int, STARVATION_END: int, IMAGING_RATE: float
    ):
        plt.vlines(x=STARVATION_START * IMAGING_RATE, ymin=0, ymax=1)
        plt.vlines(x=STARVATION_END * IMAGING_RATE, ymin=0, ymax=1)
        plt.xticks(
            np.arange(0, self.frame_of_death * IMAGING_RATE, 40), rotation=90
        )  # this line is located here because it is the first time I use IMAGING_RATE to plot something. Can be technically anywhere.

    def graph_peaks_troughs(self, IMAGING_RATE, path_to_csv):
        (self.peaks, self.troughs) = self.get_peaks_and_troughs()
        self.whi5_cycles = self.get_and_save_whi5_cycles(IMAGING_RATE, path_to_csv)
        self.paired_troughs = self.get_paired_troughs()

        plt.plot(self.x[self.peaks], self.y_scaled[self.peaks], "*")
        plt.plot(self.x[self.paired_troughs], self.y_scaled[self.paired_troughs], "v")

    def get_peaks_and_troughs(self, distance=12):
        peaks = np.array(scipy.signal.find_peaks(self.y_scaled, distance=distance)[0])
        troughs = np.array(scipy.signal.find_peaks(-self.y_scaled)[0])
        return peaks, troughs

    def get_and_save_whi5_cycles(self, IMAGING_RATE: int, path_to_csv: pathlib.Path):
        min_idx = 0
        whi5_cycles = defaultdict(list)

        for x in self.peaks:
            try:
                while x > self.troughs[min_idx]:
                    min_idx += 1
                whi5_cycles["Maxima_Index"].append(x)
                whi5_cycles["Minima_Index"].append(self.troughs[min_idx])
                whi5_cycles["Cell_ID"].append(self.id)
                whi5_cycles["Maxima_Time(min)"].append(x * IMAGING_RATE)
                whi5_cycles["Minima_Time(min)"].append(
                    self.troughs[min_idx] * IMAGING_RATE
                )
            except IndexError:
                break

        whi5_cycles = pl.DataFrame(whi5_cycles)
        path = path_to_csv / f"Whi5_CV_{id}.csv"
        whi5_cycles.write_csv(path, separator=",")

        return whi5_cycles

    def get_paired_troughs(self):
        return np.array(self.whi5_cycles["Minima_Index"])

    def graph_inflection_points(self, STARVATION_START, IMAGING_RATE):
        self.raw_inflection_points = self.find_inflection_points()
        self.export_inflection_points = self.filter_for_export_inflection_points()
        self.inflection_points, self.times_to_starvation = (
            self.filter_inflection_points_of_interest(STARVATION_START, IMAGING_RATE)
        )
        plt.vlines(
            x=[i for i in self.inflection_points],
            ymin=0,
            ymax=1,
            color="b",
        )
        for idx in range(len(self.inflection_points)):
            plt.text(
                self.inflection_points[idx] + 5,
                0.95,
                str(self.times_to_starvation[idx]),
                rotation=90,
            )

    def find_inflection_points(self):
        second_derivative = ndim.gaussian_filter1d(
            np.gradient(np.gradient(self.y_scaled)), 2
        )
        inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

        return inflection_points

    def filter_for_export_inflection_points(self):
        export_inflection_points = []
        # can be optimized
        for i in range(len(self.peaks) - 1):
            for point in self.raw_inflection_points:
                is_between_peak_and_trough = (
                    self.peaks[i] < point < self.paired_troughs[i]
                )
                if is_between_peak_and_trough:
                    export_inflection_points.append(point)

        return export_inflection_points

    def filter_inflection_points_of_interest(
        self, STARVATION_START, IMAGING_RATE, time_of_interest=120
    ):
        frames_of_interest = round(time_of_interest / IMAGING_RATE)
        inflection_points_of_interest = []
        times_to_starvation = []

        for p in self.export_inflection_points:
            frames_to_starvation = self.starvation_start - p
            is_within_time_of_interest = (
                p < self.starvation_start and frames_to_starvation < frames_of_interest
            )
            if is_within_time_of_interest:
                inflection_points_of_interest.append((self.frame_of_birth + p) * IMAGING_RATE)
                times_to_starvation.append((self.starvation_start - p) * IMAGING_RATE)

        return inflection_points_of_interest, times_to_starvation

    def graph_slope(self, IMAGING_RATE, STARVATION_START):
        self.slopes: np.ndarray = derive(self.y_scaled)
        self.y_starvation, self.slopes_starvation = self.filter_for_starvation(
            self.y_scaled, self.slopes
        )
        self.bottom_point_x, self.bottom_point_y = self.find_bottom_point(STARVATION_START)
        self.top_point_x, self.top_point_y = self.find_top_point(STARVATION_START)
        if (self.bottom_point_y != -1 and 
        self.top_point_y != -1 and
        self.bottom_point_x < self.top_point_x):
            self.slope_of_slope, self.x_of_slope, self.y_of_slope = (
                self.get_slope_of_reimport(IMAGING_RATE)
            )
            plt.plot(
                self.bottom_point_x * IMAGING_RATE,
                self.bottom_point_y,
                ".",
                color="g",
                markersize=10,
            )
            plt.plot(
                self.top_point_x * IMAGING_RATE,
                self.top_point_y,
                ".",
                color="g",
                markersize=10,
            )
            plt.plot(self.x_of_slope, self.y_of_slope, color="g", markersize=10)
            plt.text(
                self.top_point_x * IMAGING_RATE - 50,
                self.top_point_y + 0.05,
                f"Slope: {round(self.slope_of_slope[0] * 100, 4)}",
            )
            plt.text(
                self.bottom_point_x * IMAGING_RATE + 12,
                self.bottom_point_y,
                self.bottom_point_x * IMAGING_RATE,
            )

    def filter_for_starvation(self, *args):
        filtered = []
        for i in args:
            filtered.append(i[self.starvation_start : self.starvation_end])
        filtered = tuple(filtered)
        return filtered

    def find_bottom_point(self, STARVATION_START, window=4, factor=0.0005):
        for idx, y in enumerate(self.y_starvation):
            slopes_in_window = self.slopes_starvation[idx : idx + window]
            if (slopes_in_window[-1] / slopes_in_window[0]) > (
                (1 + factor / slopes_in_window[0]) ** window
            ) and all(s > 0 for s in slopes_in_window):
                point_low_x = idx + STARVATION_START
                point_low_y = y 
                return point_low_x, point_low_y
        else:
            return -1, -1

    def find_top_point(self, STARVATION_START, factor=0.8):
        for idx, y in enumerate(self.y_starvation):
            if y > factor * (max(self.y_starvation)):
                point_high_x = idx + STARVATION_START
                point_high_y = y
                return point_high_x, point_high_y
        else:
            return -1, -1

    def get_slope_of_reimport(self, IMAGING_RATE):
        x_of_slope = [
            self.bottom_point_x * IMAGING_RATE,
            self.top_point_x * IMAGING_RATE,
        ]
        y_of_slope = [self.bottom_point_y, self.top_point_y]
        slope_of_slope = np.gradient(y_of_slope, x_of_slope)
        return slope_of_slope, x_of_slope, y_of_slope

    def graph_half_reimport(self, STARVATION_START):
        try:
            last_inflection_point = round(self.inflection_points[-1] / 3)
            value_at_inflection_point = self.y_scaled[int(last_inflection_point)]
            for (idx, val) in enumerate(self.y_starvation):
                if val > value_at_inflection_point:
                    first_encounter_within_starvation = ((idx + STARVATION_START) * 3, val)
                    plt.plot(first_encounter_within_starvation[0], first_encounter_within_starvation[1], ".", markersize = 20, color = "g")
                    plt.text(first_encounter_within_starvation[0], first_encounter_within_starvation[1] - 0.05, f"val = {first_encounter_within_starvation[0]}")
                    print("plotting")
                    break
        except IndexError: # no inflection points present
            pass




    def get_individual_starvation_timing(self, STARVATION_START, STARVATION_END):
        self.starvation_start = STARVATION_START - self.frame_of_birth
        self.starvation_end = STARVATION_END - self.frame_of_death

    def save_figure(self, PATH_TO_FIGURES):
        fig = plt.gcf()
        fig.savefig(
            PATH_TO_FIGURES / f"Cell_{self.id}_Whi5_CV.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
