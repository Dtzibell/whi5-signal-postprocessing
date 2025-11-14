import pathlib
from typing import Iterable
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pybaselines import smooth
import scipy.signal
from Graphing.utils import (
    derive,
    smoothen,
    subtract_baseline,
    get_max,
    get_min,
    weigh,
)
from collections import defaultdict

"""
Does not work on cells that go missing and later reappear, but probably unnecessary?
"""

class CellGraph:
    def __init__(
        self,
        id: int,
        cell_df: pl.DataFrame,
        STARVATION_START: int,
        STARVATION_END: int,
        EXPERIMENT_LENGTH: int,
        IMAGING_RATE: float,
        CHANNEL1: str,
        CHANNEL2: str,
    ) -> None:
        """
        Initializes the CellGraph object.
        @param cell_df: pl.DataFrame; Data of cell
        Adds the following attributes to the object:
            self.STARVATION_START: int;
            self.STARVATION_END: int;
            self.EXPERIMENT_LENGTH: int;
            self.IMAGING_RATE: int;
            self.CHANNEL1: str;
            self.CHANNEL2: str;
            self.cell_df: pl.DataFrame;
            self.id: int;
            self.x: pl.Series; temporally sorted values of x in minutes
            self.y: pl.Series; temporally sorted values of y
            self.birth_frame: int; frame of cell's appearance
            self.death_frame: int; frame of cell's disappearance
            self.lifespan: int; amount of frames the cell is on screen
            self.starvation_start: int; cell's individual starvation start point. Each cell's is unique.
            self.starvation_end: int; see self.starvation_start
            self.x_growth: pl.Series; cell's x, limited to growth phase.
            self.x_starvation: pl.Series; see self.x_growth
            self.x_recovery: pl.Series; see self.x_growth
        """
        self.STARVATION_START = STARVATION_START
        self.STARVATION_END = STARVATION_END
        self.IMAGING_RATE = IMAGING_RATE
        self.EXPERIMENT_LENGTH = EXPERIMENT_LENGTH
        self.CHANNEL1 = CHANNEL1
        self.CHANNEL2 = CHANNEL2

        self.cell_df: pl.DataFrame = cell_df
        self.id: int = id

        self.x: pl.Series = self.get_x()
        self.y: pl.Series = self.get_y()

        self.birth_frame: int = np.round(min(self.x) / self.IMAGING_RATE).astype(int)
        self.death_frame: int = np.round(max(self.x) / self.IMAGING_RATE).astype(int)
        self.lifespan: int = len(self.x)

        self.starvation_start = STARVATION_START - self.birth_frame
        self.starvation_end = STARVATION_END - self.birth_frame

        self.x_growth = self.x[: self.starvation_start]
        self.x_starvation = self.x[self.starvation_start : self.starvation_end]
        self.x_recovery = self.x[self.starvation_end : self.lifespan + 1]

    def get_x(self) -> pl.Series:
        """
        Returns time points in minutes.
        """
        return self.cell_df["time_minutes"]

    def get_y(self) -> pl.Series:
        """
        Returns y signal. Converts to FRET signal and returns if CHANNEL2 is given.
        """
        if self.CHANNEL2 != "":
            y_signal_1: pl.Series = self.cell_df[self.CHANNEL1]
            y_signal_2: pl.Series = self.cell_df[self.CHANNEL2]
            return y_signal_1 / y_signal_2
        return self.cell_df[self.CHANNEL1]  # e.g. Quad2_mCherry_CV

    def initialize_figure(self, tick_interval: float = 40) -> None:
        """
        Initializes the cell's figure by constructing and designing the axes.
        @param tick_interval: float; time between two x ticks of the figure
        Adds the following attributes to the object:
            self.figure: matplotlib.figure.Figure;
            self.ax1, self.ax2, self.ax3: matplotlib.axes.Axes;
        """

        ratio_growth = self.STARVATION_START
        ratio_starvation = self.STARVATION_END - self.STARVATION_START
        ratio_recovery = self.EXPERIMENT_LENGTH - self.STARVATION_END
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1,
            3,
            sharey=True,
            gridspec_kw={ # sets the width of individual subfigures in relation to the others
                "width_ratios": [ratio_growth, ratio_starvation, ratio_recovery]
            },
        )

        self.ax1.spines["right"].set_visible(False)
        self.ax2.spines[["right", "left"]].set_visible(False)
        self.ax3.spines["left"].set_visible(False)

        self.ax1.yaxis.tick_left()
        self.ax2.tick_params(left=False)
        self.ax3.yaxis.tick_right()
        self.ax3.tick_params(labelright="off")

        d = 0.015
        kwargs = dict(transform=self.ax1.transAxes, color="k", clip_on=False)
        self.ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        self.ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        self.ax1.set_xlim(0, self.STARVATION_START * 3)

        if ratio_starvation > 0:
            kwargs = dict(transform=self.ax2.transAxes, color="k", clip_on=False)
            self.ax2.plot(
                (
                    -d * ratio_growth / ratio_starvation,
                    +d * ratio_growth / ratio_starvation,
                ),
                (1 - d, 1 + d),
                **kwargs,
            )
            self.ax2.plot(
                (
                    -d * ratio_growth / ratio_starvation,
                    +d * ratio_growth / ratio_starvation,
                ),
                (-d, +d),
                **kwargs,
            )
            self.ax2.plot(
                (
                    1 - d * ratio_growth / ratio_starvation,
                    1 + d * ratio_growth / ratio_starvation,
                ),
                (-d, +d),
                **kwargs,
            )
            self.ax2.plot(
                (
                    1 - d * ratio_growth / ratio_starvation,
                    1 + d * ratio_growth / ratio_starvation,
                ),
                (1 - d, 1 + d),
                **kwargs,
            )
            self.ax2.set_xlim(self.x_starvation[0], self.x_starvation[-1])

        if ratio_recovery > 0:
            kwargs = dict(transform=self.ax3.transAxes, color="k", clip_on=False)
            self.ax3.plot(
                (
                    -d * ratio_growth / ratio_recovery,
                    +d * ratio_growth / ratio_recovery,
                ),
                (1 - d, 1 + d),
                **kwargs,
            )
            self.ax3.plot(
                (
                    -d * ratio_growth / ratio_recovery,
                    +d * ratio_growth / ratio_recovery,
                ),
                (-d, +d),
                **kwargs,
            )
            self.ax3.set_xlim(self.x_recovery[0], self.x_recovery[-1])

        self.figure.supxlabel("Time (mins)")
        self.figure.supylabel("Norm. Whi5 CV from Full Cell Mask")
        self.figure.suptitle("Norm. Whi5 CV from Full Cell Mask")

        plt.xticks(
            np.arange(0, self.death_frame * self.IMAGING_RATE, tick_interval),
            rotation=90,
        )

    def graph_base(self) -> None:
        """
        Graphs x and y.
        Adds the following attributes to the object:
            self.y_growth, self.y_starvation, self.y_recovery: np.ndarray; y that is smoothened and scaled to 1.
        """
        # subtract_baseline fails to run if length of x_growth is 1
        smooth_y_growth = smoothen(
            self.y[: self.starvation_start], window_length=15, polyorder=3
        )
        y_growth_less_bl = subtract_baseline(self.x_growth, smooth_y_growth)
        y_starvation = smoothen(
            self.y[self.starvation_start : self.starvation_end],
            window_length=20,
            polyorder=2,
        )
        y_recovery = smoothen(self.y[self.starvation_end : len(self.x)])

        _min = get_min(smooth_y_growth, y_starvation, y_recovery)
        _max = get_max(y_growth_less_bl, y_starvation, y_recovery)
        self.y_growth: np.ndarray = y_growth_less_bl / _max
        self.y_starvation: np.ndarray = (y_starvation - _min) / _max
        self.y_recovery: np.ndarray = (y_recovery - _min) / _max

        plt.xlim(0, self.EXPERIMENT_LENGTH)
        plt.ylim(-0.1, 1.1)
        self.ax1.plot(self.x_growth, self.y_growth, "k")
        self.ax2.plot(self.x_starvation, self.y_starvation, "k")
        self.ax3.plot(self.x_recovery, self.y_recovery, "k")

    def graph_peaks_troughs(self, SINGLE_CSV_SAVING_DIR: pathlib.Path) -> None:
        """
        Saves whi5 peaks and troughs in a csv and plots.
        @param SINGLE_CSV_SAVING_DIR: pathlib.Path; path where single cell whi5 cycles are saved.
        Adds the following attributes to the object:
            self.peaks: np.ndarray; peaks of whi5
            self.troughs: np.ndarray; troughs of whi5
            self.paired_troughs: np.ndarray;
            self.whi5_cycles: pl.DataFrame; Data of Whi5 cycling
        """
        self.peaks: np.ndarray = self.get_peaks()
        self.troughs: np.ndarray = self.get_troughs()
        self.whi5_cycles: pl.DataFrame = self.save_whi5_cycles(SINGLE_CSV_SAVING_DIR)
        self.paired_troughs: pl.Series = self.get_paired_troughs()
        self.ax1.plot(self.x_growth[self.peaks], self.y_growth[self.peaks], "^")
        self.ax1.plot(
            self.x_growth[self.paired_troughs], self.y_growth[self.paired_troughs], "v"
        )

    def get_peaks(self) -> np.ndarray:
        """
        finds and returns whi5 peaks.
        @param distance: int; smallest amount of frames between two peaks
        """
        peaks: np.ndarray = np.array(
            scipy.signal.find_peaks(self.y_growth, prominence=0.15)[0]
        )
        # proms = scipy.signal.peak_prominences(self.y_growth, peaks)[0]
        # contour_heights = self.y_growth[peaks] - proms
        # self.ax1.vlines(x=(peaks + self.birth_frame)*3, ymax = self.y_growth[peaks], ymin = contour_heights)
        return peaks

    def get_troughs(self) -> np.ndarray:
        """
        Finds and returns whi5 troughs.
        """
        troughs: np.ndarray = np.array(
            scipy.signal.find_peaks(-self.y_growth, prominence=0.02)[0]
        )
        # extra trough is needed in case no trough is found after last peak
        troughs = np.append(troughs, self.starvation_start - 1)

        # proms = scipy.signal.peak_prominences(-self.y_growth, troughs)[0]
        # contour_heights = self.y_growth[troughs] + proms
        # self.ax1.vlines(x=(troughs+self.birth_frame)*3, ymin = self.y_growth[troughs], ymax = contour_heights)

        return troughs

    def save_whi5_cycles(self, SINGLE_CSV_SAVING_DIR: pathlib.Path) -> pl.DataFrame:
        """
        Pairs troughs to previously occuring peaks. Saves into csv.
        returns a pl.DataFrame of peaks and troughs 
        @param self.IMAGING_RATE: float; imaging rate in minutes
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
                cycler["Maxima_Time(min)"].append(x * self.IMAGING_RATE)
                cycler["Minima_Time(min)"].append(
                    self.troughs[min_idx] * self.IMAGING_RATE
                )
            except IndexError:
                break

        whi5_cycles: pl.DataFrame = pl.DataFrame(cycler)
        path: pathlib.Path = SINGLE_CSV_SAVING_DIR / f"Whi5_CV_{self.id}.csv"
        whi5_cycles.write_csv(path, separator=",")

        return whi5_cycles

    def get_paired_troughs(self) -> pl.Series:
        """
        Gets troughs that are preceded by a peak
        """
        try:
            return self.whi5_cycles["Minima_Index"]
        except pl.exceptions.ColumnNotFoundError:
            return pl.Series([0])

    def graph_whi5_exports(self) -> None:
        """
        Finds whi5 exports and graphs them.
        @param self.IMAGING_RATE: int; imaging rate in minutes
        Adds the following attributes to the object:
            self.exports_of_interest: list[int]; whi5 exports that lay with time_of_interest from STARVATION_START in minutes
            self.time_to_starvation: list[float]; time until STARVATION_START in minutes
            self.slopes_growth: np.ndarray; first derivative of y signal
        """
        raw_inflection_points: np.ndarray = self.find_inflection_points()
        whi5_exports: list[int] = self.filter_whi5_exports(raw_inflection_points)
        exports_of_interest, times_to_starvation = self.filter_within_time(
            whi5_exports, 150
        )

        exports: list[float] = []
        times:list[float] = []
        for i in range(len(exports_of_interest)):
            if self.peaks[-1] < exports_of_interest[i] < self.troughs[-1]:
                exports.append(exports_of_interest[i])
                times.append(times_to_starvation[i])

        self.slopes_growth:np.ndarray = derive(self.y_growth, 1)
        # self.ax1.plot(self.x_growth, self.slopes_growth, "--")
        slopes_paired = self.slopes_growth[np.round(exports).astype(int)]

        if slopes_paired.size > 0 and np.max(slopes_paired) != 0:
            self.export: int = weigh(exports, slopes_paired)
            self.time_to_starvation: float = weigh(times, slopes_paired)
            self.ax1.vlines(
                x=(self.birth_frame + self.export) * 3,
                ymin=0,
                ymax=1,
                color="b",
            )
            self.ax1.text(
                (self.birth_frame + self.export) * 3 + 5,
                0.95,
                str(round((self.starvation_start - self.time_to_starvation) * 3)),
                rotation=90,
            )


    def find_inflection_points(self) -> np.ndarray:
        """
        Gets inflection points
        """
        second_derivative: np.ndarray = derive(self.y_growth, 2)
        inflection_points: np.ndarray = np.where(np.diff(np.sign(second_derivative)))[0]
        # self.ax1.vlines((inflection_points + self.birth_frame)*3, ymin=0, ymax=1, color="g")
        return inflection_points

    def filter_whi5_exports(self, iter: Iterable) -> list[int]:
        """
        filters an iterable for items that are between peaks and troughs.
        @param iter: Iterable;
        """
        items_between_peak_and_trough: list[int] = []
        for i in range(len(self.peaks)):
            for item in iter:
                try:
                    is_between_peak_and_trough: bool = (
                        self.peaks[i] < item < self.paired_troughs[i]
                    )
                    if is_between_peak_and_trough:
                        items_between_peak_and_trough.append(item)
                except IndexError:
                    continue
        return items_between_peak_and_trough

    def filter_within_time(self, iter: Iterable, time_of_interest: int):
        """
        Filter an iterable for items that lay within time_of_interest to STARVATION_START
        @param iter: Iterable;
        @param time_of_interest: int;
        @param self.IMAGING_RATE: float, imaging rate in minutes
        """
        frames_of_interest: int = round(time_of_interest / self.IMAGING_RATE)
        inflection_points_of_interest: list[float] = []
        times_to_starvation: list[float] = []

        for p in iter:
            frames_to_starvation = self.starvation_start - p
            if 0 < frames_to_starvation < frames_of_interest:
                inflection_points_of_interest.append(p)
                times_to_starvation.append(p)

        return inflection_points_of_interest, times_to_starvation

    def graph_slope(self):
        """
        Retrieves the reimport onset, the end of reimport and the slope between those two points and plots them.
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
        # for some reason a tuple cannot consist of 1 element, so a second one is automatically added. 0th index deals with it
        self.slopes_starvation = derive(self.y_starvation, 1)
        self.reimport_onset_x, self.reimport_onset_y = self.find_reimport_onset(
        )
        self.end_of_reimport_x, self.end_of_reimport_y = self.find_end_of_reimport(
        )
        if (
            self.reimport_onset_y != -1
            and self.end_of_reimport_y != -1
            and self.reimport_onset_x < self.end_of_reimport_x
        ):
            self.slope_of_slope, self.x_of_slope, self.y_of_slope = (
                self.get_slope_of_reimport()
            )
            self.ax2.plot(
                self.reimport_onset_x * self.IMAGING_RATE,
                self.reimport_onset_y,
                ".",
                color="g",
                markersize=10,
            )
            self.ax2.plot(
                self.end_of_reimport_x * self.IMAGING_RATE,
                self.end_of_reimport_y,
                ".",
                color="g",
                markersize=10,
            )
            self.ax2.plot(self.x_of_slope, self.y_of_slope, color="g", markersize=10)
            self.ax2.text(
                self.end_of_reimport_x * self.IMAGING_RATE - 50,
                self.end_of_reimport_y + 0.05,
                f"Slope: {round(self.slope_of_slope[0] * 100, 4)}",
            )
            self.ax2.text(
                self.reimport_onset_x * self.IMAGING_RATE + 12,
                self.reimport_onset_y,
                self.reimport_onset_x * self.IMAGING_RATE,
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

    def find_reimport_onset(
        self, window: int = 4, factor: float = 0.0005
    ) -> tuple[int, float]:
        """
        Finds the frame and the signal value where the cell starts whi5 reimport.
        @param window: int; amount of points to consider. Higher values are more sensitive to slow reimport.
        @param factor: float; the amount that a series of slopes must have increased for point 0 to be recognized as reimport onset. The larger it is, the more the slopes have to increase with each step.
        """
        for idx, y in enumerate(self.y_starvation):
            slopes_in_window = self.slopes_starvation[idx : idx + window]
            if (slopes_in_window[-1] / slopes_in_window[0]) > (
                (1 + factor / slopes_in_window[0]) ** window
            ) and all(s > 0 for s in slopes_in_window):
                reimport_onset_x = idx + self.STARVATION_START
                reimport_onset_y = y
                return reimport_onset_x, reimport_onset_y
        else:
            return -1, -1

    def find_end_of_reimport(
        self, factor: float = 0.8
    ) -> tuple[int, float]:
        """
        Finds the first point that is larger than the maximum y in starvation multiplied by a factor
        @param factor: float; the number that the maximum y is multiplied by. The higher it is, the higher the point will be set.
        """
        for idx, y in enumerate(self.y_starvation):
            if y > factor * (max(self.y_starvation)):
                end_of_reimport_x: int = idx + self.STARVATION_START
                end_of_reimport_y: float = y
                return end_of_reimport_x, end_of_reimport_y
        else:
            return -1, -1

    def get_slope_of_reimport(self) -> tuple[np.ndarray, list[float], list[float]]:
        """
        Finds the slope of whi5 reimport.
        """
        x_of_slope: list[float] = [
            self.reimport_onset_x * self.IMAGING_RATE,
            self.end_of_reimport_x * self.IMAGING_RATE,
        ]
        y_of_slope: list[float] = [self.reimport_onset_y, self.end_of_reimport_y]
        slope_of_slope: np.ndarray = np.gradient(y_of_slope, x_of_slope)
        return slope_of_slope, x_of_slope, y_of_slope

    def graph_half_reimport(self):
        try:
            value_at_inflection_point = self.y_growth[self.export - self.birth_frame]
            for idx, val in enumerate(self.y_starvation):
                if val > value_at_inflection_point:
                    first_encounter_within_starvation = (
                        (idx + self.STARVATION_START) * 3,
                        val,
                    )
                    self.ax2.plot(
                        first_encounter_within_starvation[0],
                        first_encounter_within_starvation[1],
                        ".",
                        markersize=20,
                        color="g",
                    )
                    self.ax2.text(
                        first_encounter_within_starvation[0],
                        first_encounter_within_starvation[1] - 0.05,
                        f"val = {first_encounter_within_starvation[0]}",
                    )
                    break
        except IndexError:  # no inflection points present
            pass
        except AttributeError:
            pass

    def save_figure(self, PATH_TO_FIGURES: pathlib.Path):
        """
        Saves the figure.
        @param PATH_TO_FIGURES: pathlib.Path;
        """
        self.figure.savefig(
            PATH_TO_FIGURES / f"Cell_{self.id}_Whi5_CV.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
