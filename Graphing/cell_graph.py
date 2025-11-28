import pathlib
from typing import Iterable
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scipy.signal
from Graphing.utils import (
    derive,
    normalize,
    smoothen,
    subtract_baseline,
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
        slope_multiplier: float,
        slope_index: int,
    ) -> None:
        """
        Initializes the CellGraph object.
        @param id: int; ID (number) of the cell
        @param cell_df: pl.DataFrame; Cell cycling data
        @param STARVATION_START: int;
        @param STARVATION_END: int;
        @param EXPERIMENT_LENGTH: int;
        @param IMAGING_RATE: float;
        @param CHANNEL1: str;
        @param CHANNEL2: str;
        @param slope_multiplier: float; Values between 0 and 1; Refer to
        look_forward for description
        @param slope_index: int; Refer to look_forward for description
        Adds the following attributes to the object:
            self.STARVATION_START: int;
            self.STARVATION_END: int;
            self.EXPERIMENT_LENGTH: int;
            self.IMAGING_RATE: int;
            self.CHANNEL1: str;
            self.CHANNEL2: str; Second fluorescent channel of the experiment;
            present in FRET setups.
            self.GROWTH_INCREMENT: int; Amount of frames past STARVATION_START
            to plot in growth phase.
            self.STARVATION_DECREMENT: int; Amount of frames before
            STARVATION_START to plot in starvation phase.
            self.cell_df: pl.DataFrame;
            self.id: int;
            self.x: np.ndarray; temporally sorted values of x in minutes
            self.y: np.ndarray; temporally sorted values of y
            self.birth_frame: int; frame of cell's appearance
            self.death_frame: int; frame of cell's disappearance
            self.lifespan: int; amount of frames the cell is on screen
            self.starvation_start: int; cell's individual starvation start point. Each cell's is unique.
            self.starvation_end: int; see self.starvation_start
            self.x_growth: np.ndarray; time within growth phase. Starts from
            self.birth_frame * self.IMAGING_RATE.
            self.x_starvation: np.ndarray; time within starvation phase.
            self.x_recovery: np.ndarray; time within recovery phase. Ends at
            self.lifespan.
            self.slope_multiplier: int; Refer to look_forward for description
            self.slope_index: int; Refer to look_forward for description
        """
        self.STARVATION_START = STARVATION_START
        self.STARVATION_END = STARVATION_END
        self.IMAGING_RATE = IMAGING_RATE
        self.EXPERIMENT_LENGTH = EXPERIMENT_LENGTH
        self.CHANNEL1 = CHANNEL1
        self.CHANNEL2 = CHANNEL2
        self.GROWTH_INCREMENT = 5
        self.STARVATION_DECREMENT = 8

        self.cell_df: pl.DataFrame = cell_df
        self.id: int = id

        self.x: np.ndarray = self.get_x()
        self.y: np.ndarray = self.get_y()

        self.birth_frame: int = np.round(min(self.x) / self.IMAGING_RATE).astype(int)
        self.death_frame: int = np.round(max(self.x) / self.IMAGING_RATE).astype(int)
        self.lifespan: int = self.x.size

        self.starvation_start = STARVATION_START - self.birth_frame
        self.starvation_end = STARVATION_END - self.birth_frame

        self.x_growth = self.x[: self.starvation_start + self.GROWTH_INCREMENT]
        self.x_starvation = self.x[
            self.starvation_start - self.STARVATION_DECREMENT : self.starvation_end
        ]
        self.x_recovery = self.x[self.starvation_end : self.lifespan + 1]

        self.slope_multiplier = slope_multiplier
        self.slope_index = slope_index + 1

    def get_x(self) -> np.ndarray:
        """
        Returns time points in minutes.
        """
        return self.cell_df["time_minutes"].to_numpy()

    def get_y(self) -> np.ndarray:
        """
        Returns y signal. Converts to FRET signal and returns if CHANNEL2 is given.
        """
        if self.CHANNEL2 != "":
            y_signal_1: np.ndarray = self.cell_df[self.CHANNEL1].to_numpy()
            y_signal_2: np.ndarray = self.cell_df[self.CHANNEL2].to_numpy()
            return y_signal_1 / y_signal_2
        return self.cell_df[self.CHANNEL1].to_numpy()  # e.g. Quad2_mCherry_CV

    def time_at_(self, frame):
        """
        Returns the time point at frame.
        """
        return self.x[frame]

    def growth_time_at(self, frame):
        """
        Returns the time point at frame of x_growth.
        """
        return self.x_growth[frame]

    def starvation_time_at_(self, frame):
        return self.x_starvation[frame]

    def growth_signal_at_(self, frame):
        return self.y_growth[frame]

    def starvation_signal_at_(self, frame):
        return self.y_starvation[frame]

    def recovery_signal_at(self, frame):
        return self.y_recovery[frame]

    def normalized_signal_at_(self, frame):
        return self.y_normalized[frame]

    def initialize_figure(self, tick_interval: float = 40) -> None:
        """
        Initializes the cell's figure by constructing and designing the axes.

        This function creates a plot that has 3 separated segments with a
        shared x axis, one segment per experiment's stage.
        The y axis flanks the graph's vertical edges.

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
            gridspec_kw={  # sets the width of individual subfigures in relation to the others
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
            self.y_growth: np.ndarray; y within growth phase. Dimensions
            correspond to self.x_growth. Smoothened, baseline removed and
            normalized to 1.
            self.y_starvation: np.ndarray; y within starvation phase. Dimensions
            correspond to self.x_starvation. Smoothened, interval minimum y
            removed and normalized to 1.
            self.y_recovery: np.ndarray; y within recovery phase. Only applies
            if experiment extends into recovery phase. Dimensions correspond
            self.x_recovery. Smoothened, baseline removed and normalized to 1.
            self.y_normalized: np.ndarray; y throughout the experiment.
            Dimensions correspond to self.x. smoothened, minimum removed and
            normalized to 1.
        """
        # subtract_baseline fails to run if length of x_growth is 1
        window_length = 10
        polyorder = 2
        y_full = smoothen(self.y, window_length=window_length, polyorder=polyorder)
        y_growth = smoothen(
            self.y[: self.starvation_start + self.GROWTH_INCREMENT],
            window_length=window_length,
            polyorder=polyorder,
        )
        y_growth_less_bl = subtract_baseline(self.x_growth, y_growth)
        y_starvation = smoothen(
            self.y[
                self.starvation_start - self.STARVATION_DECREMENT : self.starvation_end
            ],
            window_length=window_length,
            polyorder=3,
        )
        if self.starvation_end + self.birth_frame > self.STARVATION_END:
            y_recovery = smoothen(
                self.y[self.starvation_end : self.lifespan + 1],
                window_length=window_length,
                polyorder=polyorder,
            )
            y_recovery_less_bl = subtract_baseline(self.x_recovery, y_recovery)
            self.y_recovery: np.ndarray = normalize(y_recovery_less_bl)
            self.ax3.plot(self.x_recovery, self.y_recovery, "k", "--")
            self.ax3.plot(self.x, y_full, "k")

        self.y_normalized: np.ndarray = normalize(y_full)
        self.y_growth: np.ndarray = normalize(y_growth_less_bl)
        self.y_starvation: np.ndarray = normalize(y_starvation)

        plt.ylim(-0.1, 1.1)
        self.ax1.plot(self.x, self.y_normalized, c="k")
        self.ax2.plot(self.x, self.y_normalized, c="k")
        self.ax1.plot(self.x_growth, self.y_growth, ls="--", c="k", alpha=0.3)
        self.ax2.plot(self.x_starvation, self.y_starvation, ls="--", c="k", alpha=0.3)

    def graph_peaks_troughs(self, SINGLE_CSV_SAVING_DIR: pathlib.Path) -> None:
        """
        Saves whi5 peaks and troughs in a csv and plots.
        @param SINGLE_CSV_SAVING_DIR: pathlib.Path; path where single cell whi5 
        cycles are saved.
        Adds the following attributes to the object:
            self.peaks: np.ndarray; peaks of whi5
            self.troughs: np.ndarray; troughs of whi5
            self.whi5_cycles: pl.DataFrame; Data of Whi5 cycling
            self.paired_troughs: np.ndarray; troughs following peaks.
        """
        self.peaks: np.ndarray = self.get_peaks()
        self.troughs: np.ndarray = self.get_troughs()
        self.whi5_cycles: pl.DataFrame = self.save_whi5_cycles(SINGLE_CSV_SAVING_DIR)
        self.paired_troughs: np.ndarray = self.get_paired_troughs()
        self.ax1.plot(self.x[self.peaks], self.y_normalized[self.peaks], "^")
        self.ax1.plot(
            self.x[self.paired_troughs], self.y_normalized[self.paired_troughs], "v"
        )

    def get_peaks(self) -> np.ndarray:
        """
        Returns beginning of Whi5 export.
        @param distance: int; smallest amount of frames between two peaks
        """
        peaks: np.ndarray = np.array(
            scipy.signal.find_peaks(
                self.y_growth[: -self.GROWTH_INCREMENT], prominence=0.2
            )[0]
        )
        # proms = scipy.signal.peak_prominences(self.y_growth, peaks)[0]
        # contour_heights = self.y_growth[peaks] - proms
        # self.ax1.vlines(x=(peaks + self.birth_frame)*3, ymax = self.y_growth[peaks], ymin = contour_heights)
        return peaks

    def get_troughs(self) -> np.ndarray:
        """
        Returns end of Whi5 export.
        Adds the following attributes to the object:
            self.d1y_growth: np.ndarray; first derivative of y_growth
        """
        self.d1y_growth: np.ndarray = derive(self.y_growth, 1)
        raw_troughs = np.nonzero(np.diff(np.sign(self.d1y_growth)))[0]
        troughs = self.look_forward(raw_troughs)
        # extra trough is needed in case no trough is found after last peak
        troughs = np.append(troughs, self.starvation_start - 1)
        return troughs

    def look_forward(self, troughs: np.ndarray) -> np.ndarray:
        """
        Removes troughs if they are located within the whi5 export interval.
        """
        filtered_troughs = []
        slopes_min = np.min(self.d1y_growth)
        for t in troughs:
            if not np.any(
                self.d1y_growth[t + 1 : t + self.slope_index]
                < slopes_min * self.slope_multiplier
            ):
                filtered_troughs.append(t)
        return np.round(np.array(filtered_troughs)).astype(int)

    def save_whi5_cycles(self, SINGLE_CSV_SAVING_DIR: pathlib.Path) -> pl.DataFrame:
        """
        Pairs troughs to previously occuring peaks. Saves peaks and troughs into
        csv.
        Returns pl.DataFrame of peaks and troughs
        @param SINGLE_CSV_SAVING_DIR: pathlib.Path; path where single cell whi5 cycles are saved.
        """
        min_idx: int = 0
        cycler: defaultdict = defaultdict(list)

        for x in self.peaks:
            try:
                while x + 2 > self.troughs[min_idx]:
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

    def get_paired_troughs(self) -> np.ndarray:
        """
        Gets paired troughs
        """
        try:
            return self.whi5_cycles["Minima_Index"].to_numpy()
        except pl.exceptions.ColumnNotFoundError:
            return np.array([0])

    def graph_whi5_exports(self) -> None:
        """
        Graphs the last Whi5 export.
        """
        if self.peaks.size > 0:
            d1y_last_peak = self.d1y_growth[self.peaks[-1] : self.paired_troughs[-1]]
            average_inflp = self.average_inflp(
                self.x_growth[self.peaks[-1] : self.paired_troughs[-1]], d1y_last_peak
            )
            self.ax1.vlines(average_inflp, ymin=0, ymax=1, color="b")
            self.ax1.text(
                average_inflp + 5,
                0.95,
                str(round(average_inflp, 2)),
                rotation=90,
            )

    def graph_half_reimport(self) -> None:
        """
        Finds the Whi5 reimport start and end, the inflection point between them
        and plots.
        """
        trough: int = self.get_reimport_onset(factor=0.005)
        peak: int = self.get_reimport_peak(0.6)
        if trough >= 0 and peak >= 0:
            inflection_point = self.average_inflp(
                self.x_starvation[trough : peak + 1],
                derive(self.y_starvation[trough : peak + 1], 1),
            )
            self.ax2.plot(
                self.time_at_(self.starvation_start + peak - self.STARVATION_DECREMENT),
                self.normalized_signal_at_(
                    self.starvation_start + peak - self.STARVATION_DECREMENT
                ),
                "+",
            )
            self.ax2.plot(
                self.x[self.starvation_start + trough - self.STARVATION_DECREMENT],
                self.normalized_signal_at_(
                    self.starvation_start + trough - self.STARVATION_DECREMENT
                ),
                "+",
            )

            self.ax2.vlines(
                inflection_point,
                ymin=0,
                ymax=1,
                color="b",
            )
            self.ax2.text(
                inflection_point + 5,
                0.95,
                str(int(round(inflection_point, 2))),
                rotation=90,
            )

    def average_inflp(
        self, x_between_peaks: np.ndarray, d1y_between_peaks: np.ndarray
    ) -> int:
        """
        Approximates the inflection point of a segment.
        Returns the time at inflection.
        """
        weights = d1y_between_peaks / np.sum(d1y_between_peaks)
        cum_weights = np.cumsum(weights)
        average_infl = np.searchsorted(cum_weights, 0.5)
        return x_between_peaks[average_infl]

    def get_reimport_peak(self, factor: float) -> int:
        """
        Gets the approximate end of Whi5 reimport during starvation.
        """
        peaks: np.ndarray = (
            scipy.signal.find_peaks(self.y_starvation, prominence=0.03)
        )[0]
        for p in peaks:
            if self.starvation_signal_at_(p) > factor:
                return p
        return -1

    def get_reimport_onset(self, factor: float) -> int:
        """
        Gets the approximate start of Whi5 reimport during starvation.
        Adds the following attributes to the object:
            self.d1y_starvation: np.ndarray; the derivative of self.y_starvation.
        """
        interval = 5
        if self.y_starvation.size > 1:
            self.d1y_starvation = derive(self.y_starvation, 1)
            #    self.ax2.plot(self.x_starvation, self.d1y_starvation, ls="--", c="b")
            for idx in range(self.y_starvation.size - interval):
                if (
                    self.starvation_signal_at_(idx + interval)
                    > (1 + factor * self.starvation_signal_at_(idx)) ** interval
                    * self.starvation_signal_at_(idx)
                    and np.diff(self.y_starvation[idx : idx + 2]) > 0
                ):
                    return idx
        return -1

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
