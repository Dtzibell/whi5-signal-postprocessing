import pathlib
import numpy as np
import polars as pl
from polars import col as c
import matplotlib.pyplot as plt
import scipy.signal
from Graphing.utils import (
    derive,
    normalize,
    smoothen,
    subtract_baseline,
)
from collections import defaultdict

class Lifespan:
    """
    A container class for a cells lifespan
    The values are stored in frames
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.duration = end - start


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
        self.cell_df: pl.DataFrame = cell_df
        self.id: int = id
        self.STARVATION_START: int = STARVATION_START
        self.STARVATION_END: int = STARVATION_END
        self.IMAGING_RATE: float = IMAGING_RATE
        self.EXPERIMENT_LENGTH: int = EXPERIMENT_LENGTH
        self.CHANNEL1: str = CHANNEL1
        self.CHANNEL2: str = CHANNEL2

        # motivation for these variables: the last Whi5 export
        # is not accurate if it goes outside the bounds of the
        # pre-starvation graph. The same applies to starvation
        # phase, where a Whi5 reimport is looked for. The shift
        # likely happens because of smoothing of the graph.
        # Values tend to shift.
        self.GROWTH_INCREMENT: int = 5
        self.STARVATION_DECREMENT: int = 5

        self.cell_df: pl.DataFrame = self.get_y()

        self.lifespan: Lifespan = Lifespan(
            self.cell_df[0, "frame_i"], self.cell_df[-1, "frame_i"]
        )

        self.has_starvation = (
            self.cell_df.filter(
                c("frame_i").is_in(range(self.STARVATION_START, self.STARVATION_END))
            ).height
            > 0
        )
        self.has_poststarvation = (
            self.cell_df.filter(
                c("frame_i").is_in(range(self.STARVATION_END, self.lifespan.end))
            ).height
            > 0
        )

        self.slope_multiplier: float = slope_multiplier
        self.slope_index: int = slope_index + 1

    def get_x(self) -> pl.Series:
        """
        Returns time points in minutes.
        """
        return self.cell_df.get_column("time_minutes")

    def get_y(self) -> pl.DataFrame:
        """
        Returns y signal. Converts to FRET signal and returns if CHANNEL2 is given.
        """
        match self.CHANNEL2:
            case "":
                return self.cell_df.rename({self.CHANNEL1: "signal"})
            case str():
                y_signal_1: pl.Series = self.cell_df.get_column(self.CHANNEL1)
                y_signal_2: pl.Series = self.cell_df.get_column(self.CHANNEL2)
                return self.cell_df.with_columns(signal=y_signal_1 / y_signal_2)

    def get_prestarvation_df(self):
        return self.cell_df.filter(c("time_minutes") < self.STARVATION_START)

    def get_starvation_df(self):
        return self.cell_df.filter(
            self.STARVATION_END > c("time_minutes") > self.STARVATION_START
        )

    def get_poststarvation_df(self):
        return self.cell_df.filter(c("time_minutes") > self.STARVATION_END)

    def get_birth_frame(self):
        return self.cell_df[0, "frame_i"]

    def get_death_frame(self):
        return self.cell_df[-1, "frame_i"]

    def time_at_(self, frames: pl.Series | int):
        """
        Returns the time in minutes at frames.
        """

        match frames:
            case pl.Series():
                return (
                    self.cell_df.lazy()
                    .select(["frame_i", "time_minutes"])
                    .filter(c("frame_i").is_in(frames))
                    .collect()
                    .get_column("time_minutes")
                )
            case int():
                return self.cell_df[frames, "time_minutes"]

    def initialize_figure(self) -> None:
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
        # Experiment length seems to end with a 9, meanwhile starvation
        # end is assumed to end with a 0. So add 1
        ratio_recovery = self.EXPERIMENT_LENGTH - self.STARVATION_END + 1
        ratios = [r for r in [ratio_growth, ratio_starvation, ratio_recovery] if r > 0]
        self.figure, self.axes = plt.subplots(
            1,
            len(ratios),
            # sharey=True,
            gridspec_kw={  # sets the width of individual subfigures in relation to the others
                "width_ratios": ratios
            },
        )
        for i, a in enumerate(self.axes):
            setattr(self, f"ax{i + 1}", a)

    def graph_full(self) -> None:
        window_length: int = 10
        polyorder: int = 2
        self.cell_df: pl.DataFrame = smoothen(
            self.cell_df, window_length=window_length, polyorder=3
        )
        self.cell_df: pl.DataFrame = normalize(self.cell_df, "signal_smooth")
        x = self.cell_df["time_minutes"]
        y = self.cell_df["signal_normalized"]
        for ax in self.axes:
            ax.plot(x, y, c="k")

    def graph_prestarvation(self) -> None:
        if hasattr(self, "ax1"):
            self.prestarvation: pl.DataFrame = self.cell_df.filter(
                c("frame_i").is_in(
                    range(0, self.STARVATION_START + self.GROWTH_INCREMENT)
                )
            )
            self.prestarvation = subtract_baseline(
                self.prestarvation, "frame_i", "signal_smooth"
            )
            self.prestarvation: pl.DataFrame = normalize(
                self.prestarvation, "signal_less_bl"
            )
            x = self.prestarvation["time_minutes"]
            y = self.prestarvation["signal_normalized"]
            self.ax1.plot(
                x,
                y,
                ls="--",
                c="k",
                alpha=0.3,
            )
        else:
            print(f"WARNING: Cell {self.id} has no prestarvation phase")

    def graph_starvation(self) -> None:
        if hasattr(self, "ax2"):
            self.starvation: pl.DataFrame = self.cell_df.filter(
                c("frame_i").is_in(
                    range(
                        self.STARVATION_START - self.STARVATION_DECREMENT,
                        self.STARVATION_END,
                    )
                )
            )
            self.starvation: pl.DataFrame = normalize(self.starvation, "signal_smooth")
            x = self.starvation["time_minutes"]
            y = self.starvation["signal_normalized"]
            self.ax2.plot(
                x,
                y,
                ls="--",
                c="k",
                alpha=0.3,
            )
        else:
            print(f"WARNING: Cell {self.id} has no starvation phase")

    def graph_poststarvation(self) -> None:
        if hasattr(self, "ax3"):
            self.poststarvation: pl.DataFrame = self.cell_df.filter(
                c("frame_i").is_in(range(self.STARVATION_END, self.lifespan.end))
            )
            self.poststarvation = subtract_baseline(
                self.poststarvation, "frame_i", "signal_smooth"
            )
            self.poststarvation = normalize(self.poststarvation, "signal_less_bl")
            self.ax3.plot(
                self.poststarvation["time_minutes"],
                self.poststarvation["signal_normalized"],
                ls="--",
                c="k",
                alpha=0.3,
            )
        else:
            print(f"WARNING: Cell {self.id} has no poststarvation phase")

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
        if hasattr(self, "ax1"):
            self.peaks: np.ndarray = self.get_peaks()
            self.troughs: np.ndarray = self.get_troughs()
            self.paired_troughs: np.ndarray = self.get_paired_troughs()
            self.whi5_cycles: pl.DataFrame = self.save_whi5_cycles(
                SINGLE_CSV_SAVING_DIR
            )
            if self.id == 25:
                print(self.prestarvation, self.peaks, self.paired_troughs)
            self.ax1.plot(
                self.cell_df[self.peaks, "time_minutes"],
                self.cell_df[self.peaks, "signal_normalized"],
                "^",
            )
            self.ax1.plot(
                self.cell_df[self.paired_troughs, "time_minutes"],
                self.cell_df[self.paired_troughs, "signal_normalized"],
                "v",
            )

    def get_peaks(self) -> np.ndarray:
        """
        Returns beginning of Whi5 export.
        @param distance: int; smallest amount of frames between two peaks
        """
        pre_wo_increment = self.prestarvation  # [: -self.GROWTH_INCREMENT]
        peaks: np.ndarray = np.array(
            scipy.signal.find_peaks(
                pre_wo_increment["signal_normalized"], prominence=0.2
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
        self.prestarvation: pl.DataFrame = derive(self.prestarvation, 1)
        raw_troughs = np.nonzero(np.diff(np.sign(self.prestarvation["signal_d1y"])))[0]
        troughs = self.look_forward(raw_troughs)
        # extra trough is needed in case no trough is found after last peak
        troughs = np.append(troughs, self.STARVATION_START - self.lifespan.start - 1)
        return troughs

    def look_forward(self, troughs: np.ndarray) -> np.ndarray:
        """ """
        d1y = self.prestarvation["signal_d1y"].to_numpy()
        filtered_troughs = []
        slopes_min = np.min(d1y)
        for t in troughs:
            # Supposed to find real troughs.
            # Checks the next slope_index points
            # to see if they are not bigger than slopes_min * slope_multiplier.
            # This way, the troughs that are followed by steep downward slopes
            # are removed.
            if not np.any(
                d1y[t + 1 : t + self.slope_index] < slopes_min * self.slope_multiplier
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
        cycler: defaultdict = defaultdict(list)

        for i, p in enumerate(self.peaks):
            if i >= self.paired_troughs.size:
                break
            cycler["Cell_ID"].append(self.id)
            cycler["Maxima_Index"].append(p)
            cycler["Minima_Index"].append(self.paired_troughs[i])
            cycler["Maxima_Time(min)"].append(p * self.IMAGING_RATE)
            cycler["Minima_Time(min)"].append(self.troughs[i] * self.IMAGING_RATE)

        whi5_cycles: pl.DataFrame = pl.DataFrame(cycler)
        path: pathlib.Path = SINGLE_CSV_SAVING_DIR / f"Whi5_CV_{self.id}.csv"
        whi5_cycles.write_csv(path)

        return whi5_cycles

    def get_paired_troughs(self) -> np.ndarray:
        """
        Gets paired troughs
        """
        paired_troughs = []
        trough_index = 0
        troughs_size = len(self.troughs)
        for i in range(self.peaks.size):
            while True:
                if trough_index >= troughs_size:
                    self.peaks = self.peaks[:i]
                    break
                if self.troughs[trough_index] > self.peaks[i]:
                    paired_troughs.append(self.troughs[trough_index])
                    break
                trough_index += 1

        return np.array(paired_troughs)

    def graph_whi5_exports(self) -> None:
        """
        Graphs the last Whi5 export.
        """
        if hasattr(self, "ax1"):
            if self.peaks.size > 0:
                d1y = self.prestarvation["signal_d1y"]
                x = self.prestarvation["time_minutes"]
                print(self.peaks[-1], self.paired_troughs[-1])
                if self.peaks.size > 0 and self.paired_troughs.size > 0:
                    d1y_last_peak = d1y[
                        self.peaks[-1] : self.paired_troughs[-1]
                    ].to_numpy()
                    x_last_peak = x[self.peaks[-1] : self.paired_troughs[-1]].to_numpy()
                    average_inflp = self.approximate_inflp(x_last_peak, d1y_last_peak)
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
        if hasattr(self, "ax2"):
            trough: int = self.get_reimport_onset(factor=0.005)
            peak: int = self.get_reimport_peak(0.6)
            if peak > trough:
                inflection_point = self.approximate_inflp(
                    self.starvation[trough : peak + 1, "time_minutes"].to_numpy(),
                    derive(self.starvation[trough : peak + 1], 1, "signal_normalized")[
                        "time_minutes"
                    ].to_numpy(),
                )
                self.ax2.plot(
                    self.cell_df[
                        self.STARVATION_START
                        - self.lifespan.start
                        + peak
                        - self.STARVATION_DECREMENT,
                        "time_minutes",
                    ],
                    self.cell_df[
                        self.STARVATION_START
                        - self.lifespan.start
                        + peak
                        - self.STARVATION_DECREMENT,
                        "signal_normalized",
                    ],
                    "+",
                )
                self.ax2.plot(
                    self.cell_df[
                        self.STARVATION_START
                        - self.lifespan.start
                        + trough
                        - self.STARVATION_DECREMENT,
                        "time_minutes",
                    ],
                    self.cell_df[
                        self.STARVATION_START
                        - self.lifespan.start
                        + trough
                        - self.STARVATION_DECREMENT,
                        "signal_normalized",
                    ],
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

    def approximate_inflp(
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
            scipy.signal.find_peaks(
                self.starvation["signal_normalized"], prominence=0.03
            )
        )[0]
        for p in peaks:
            if self.starvation[p.item(), "signal_normalized"] > factor:
                return p.item()
        return -1

    def get_reimport_onset(self, factor: float) -> int:
        """
        Gets the approximate start of Whi5 reimport during starvation.
        """
        interval = 5
        if self.starvation.height > 1:
            sig_starv = self.starvation["signal_normalized"]
            for idx in range(self.starvation.height - interval):
                if (
                    sig_starv[idx + interval]
                    > (1 + factor * sig_starv[idx]) ** interval * sig_starv[idx]
                    and np.diff(sig_starv[idx : idx + 2]) > 0
                ):
                    return idx
        return -1

    def format_figure(self):
        plt.xticks(
            np.arange(0, self.lifespan.end * self.IMAGING_RATE, 40),
            rotation=75,
        )
        self.format_ax1()
        self.format_ax2()
        self.format_ax3()
        self.figure.supxlabel("Time (mins)")
        self.figure.supylabel("Norm. Whi5 CV from Full Cell Mask")
        self.figure.suptitle("Norm. Whi5 CV from Full Cell Mask")

    def format_ax1(self):
        if hasattr(self, "ax1"):
            self.ax1.spines["right"].set_visible(False)
            self.ax1.yaxis.tick_left()
            d = 0.015
            kwargs = dict(transform=self.ax1.transAxes, color="k", clip_on=False)
            self.ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            self.ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            self.ax1.set_xticks(
                np.arange(
                    0,
                    (self.STARVATION_START + self.GROWTH_INCREMENT) * self.IMAGING_RATE,
                    50,
                ),
            )
            self.ax1.tick_params(axis="x", labelrotation=75)
            self.ax1.set_xlim(0, (self.GROWTH_INCREMENT + self.STARVATION_START) * 3)

    def format_ax2(self):
        if hasattr(self, "ax2"):
            d = 0.015
            kwargs = dict(transform=self.ax2.transAxes, color="k", clip_on=False)
            ratio_growth = self.STARVATION_START
            ratio_starvation = self.STARVATION_END - self.STARVATION_START
            self.ax2.spines["left"].set_visible(False)
            self.ax2.tick_params(left=False, labelleft=False)
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
            self.ax2.set_xlim(
                (self.STARVATION_START - self.STARVATION_DECREMENT) * self.IMAGING_RATE,
                self.STARVATION_END * self.IMAGING_RATE,
            )
            self.ax2.set_xticks(
                np.arange(
                    (self.STARVATION_START - self.STARVATION_DECREMENT)
                    * self.IMAGING_RATE,
                    self.STARVATION_END * self.IMAGING_RATE,
                    50,
                ),
            )
            self.ax2.tick_params(axis="x", labelrotation=75)

    def format_ax3(self):
        ratio_growth = self.STARVATION_START
        ratio_starvation = self.STARVATION_END - self.STARVATION_START
        ratio_recovery = self.EXPERIMENT_LENGTH - self.STARVATION_END
        d = 0.015
        if hasattr(self, "ax3"):
            kwargs = dict(transform=self.ax3.transAxes, color="k", clip_on=False)

            self.ax3.spines["left"].set_visible(False)
            self.ax3.yaxis.tick_right()
            self.ax3.tick_params(labelright=False)
            self.ax3.set_xlim(self.STARVATION_END * self.IMAGING_RATE, self.EXPERIMENT_LENGTH * self.IMAGING_RATE)
            self.ax3.set_xticks(
                np.arange(
                    self.STARVATION_END * self.IMAGING_RATE,
                    self.EXPERIMENT_LENGTH * self.IMAGING_RATE,
                    50,
                ),
            )
            self.ax3.tick_params(axis="x", labelrotation=75)
            d = 0.015
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
            if hasattr(self, "ax2"):
                self.ax2.spines["right"].set_visible(False)
                kwargs = dict(transform=self.ax2.transAxes, color="k", clip_on=False)
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

    def save_figure(self, PATH_TO_FIGURES: pathlib.Path):
        """
        Saves the figure.
        @param PATH_TO_FIGURES: pathlib.Path;
        """
        plt.tight_layout()
        self.figure.savefig(
            PATH_TO_FIGURES / f"Cell_{self.id}_Whi5_CV.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
