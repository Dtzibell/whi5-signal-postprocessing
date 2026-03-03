import pathlib

from PyQt5.QtWidgets import (
    QCompleter,
    QMainWindow,
    QApplication,
    QPushButton,
    QComboBox,
    QGridLayout,
    QWidget,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QFileDialog,
    QCheckBox,
    QLineEdit,
    QVBoxLayout,
)  ## AP switched to PyQt5 for now
import sys
from configparser import ConfigParser


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # initialize the configuration file
        self.config = ConfigParser()
        self.config.read("config.ini")
        # initialize the configuration file

        #### initialize the window ####
        self.setWindowTitle("ACDC evaluation")
        self.window_layout = QGridLayout()

        # the layouts stretch 2 to 1 (if window expanded)
        self.window_layout.setColumnStretch(0, 1)
        self.window_layout.setColumnStretch(1, 2)

        self.input_layout = QVBoxLayout()
        self.cross_val_layout = QVBoxLayout()
        #### initialize the window ####
        
        #### create input widgets ####
        # data button
        self.select_data_file = QPushButton("Select file")
        self.select_data_file.clicked.connect(self.open_file_selection)
        self.selected_files = []

        # is starvation box
        self.has_starvation_phase = QComboBox()
        self.has_starvation_phase.addItem("Yes")
        self.has_starvation_phase.addItem("No")
        self.has_starvation_phase.setDisabled(True)

        # slope multiplier box
        self.slope_multiplier = QDoubleSpinBox()
        self.slope_multiplier.setMaximum(1)
        self.slope_multiplier.setMinimum(0)
        self.slope_multiplier.setValue(0.1)

        # slope index box
        self.slope_index = QSpinBox()
        self.slope_index.setValue(5)

        # starvation start box
        self.starvation_start_frame = QSpinBox()
        self.starvation_start_frame.setRange(0, 500)

        # starvation end box
        self.starvation_end_frame = QSpinBox()
        self.starvation_end_frame.setRange(0, 500)

        # double fluorophore box
        self.is_fret = QCheckBox()
        self.is_fret.toggled.connect(self.gen_channel_select)

        # fluorophore channels
        self.fluorophore_channel_1 = QComboBox()
        self.fluorophore_channel_1.setEditable(True)
        choices = [s.strip() for s in self.config["FLUORESCENCE"]["SignalChoices"].split(",")]
        self.fluorophore_channel_1.addItems(choices)
        self.fluorophoreCompleter = QCompleter(choices)
        self.fluorophore_channel_1.setCompleter(self.fluorophoreCompleter)

        self.fluorophore_channel_2 = QLineEdit()

        # end button
        self.end = QPushButton("OK")
        #### create input widgets ####

        #### create labels ####
        # located in cross_val layout
        self.file_path_label = QLabel("Current file paths: ")
        self.slope_index_label = QLabel("Slope index: ")
        self.slope_multiplier_label = QLabel("Slope multiplier: ")
        self.file_path = QLabel()
        self.file_path.setObjectName("path")
        self.file_path.setWordWrap(True)

        # located in input_layout
        self.starvation_label = QLabel("Starvation? (has to be set to yes)")
        self.starv_start_label = QLabel("Start of starvation:")
        self.starv_end_label = QLabel("End of starvation:")
        self.exp_length_label = QLabel("Experiment length:")
        self.imaging_rate_label = QLabel("Image acquisition rate:")
        self.is_fret_label = QLabel("FRET experiment?")
        self.fluorophore_select_label = QLabel("Fluorophore channel name:")
        #### create labels ####

        #### add widgets to input layout ####
        self.input_layout.addWidget(self.select_data_file)
        self.input_layout.addWidget(self.starvation_label)
        self.input_layout.addWidget(self.has_starvation_phase)
        self.input_layout.addWidget(self.starv_start_label)
        self.input_layout.addWidget(self.starvation_start_frame)
        self.input_layout.addWidget(self.starv_end_label)
        self.input_layout.addWidget(self.starvation_end_frame)
        self.input_layout.addWidget(self.is_fret_label)
        self.input_layout.addWidget(self.is_fret)
        self.input_layout.addWidget(self.fluorophore_select_label)
        self.input_layout.addWidget(self.fluorophore_channel_1)
        self.input_layout.addStretch()
        #### add widgets to input layout ####

        #### add widgets to cross validation layout ####

        # add empty label to correct widget shift due to button
        self.cross_val_layout.addSpacing(32)

        self.cross_val_layout.addWidget(self.file_path_label)
        self.cross_val_layout.addWidget(self.file_path)
        self.cross_val_layout.addWidget(self.slope_index_label)
        self.cross_val_layout.addWidget(self.slope_index)
        self.cross_val_layout.addWidget(self.slope_multiplier_label)
        self.cross_val_layout.addWidget(self.slope_multiplier)
        self.cross_val_layout.addStretch()
        self.cross_val_layout.addWidget(self.end)
        #### add widgets to cross validation layout ####

        #### add layouts to window_layout ####
        # add layouts to the main grid
        self.window_layout.addLayout(self.input_layout, 0, 0)
        self.window_layout.addLayout(self.cross_val_layout, 0, 1)
        #### add layouts to window_layout ####

        #### finalize central widget ####
        self.widget = QWidget()
        self.widget.setLayout(self.window_layout)
        self.setCentralWidget(self.widget)
        #### finalize central widget ####

        #### load input presets from config.ini ####
        self.starvation_start_frame.setValue(
                int(self.config["INPUT_PRESETS"]["StarvationStartFrame"]))
        self.starvation_end_frame.setValue(
                int(self.config["INPUT_PRESETS"]["StarvationEndFrame"]))
        #### load input presets from config.ini ####

    def open_file_selection(self):
        """
        Opens a file browser dialog for selection of file with data (.csv)
        :return: a string of the full path to the file
        """
        default_dir = self.config["PATHS"]["DefaultDirectory"]
        file_dialog = QFileDialog.getOpenFileNames(
            None,
            "Select your data's .csv",
            default_dir,
            "*.csv",
        )
        print(
            file_dialog
        )  # output: list: [0] list of paths as strings, 
           # [1] type of files as string (ex. "*.csv")
        for path in file_dialog[0]:
            self.selected_files.append(
                pathlib.Path(path)
            )  # creates a list of pathlib.Paths with selected files

        # for visual representation of selection
        if len(file_dialog) > 1:
            if len(file_dialog[0]) > 1:
                file_names = ""
                for i in range(len(file_dialog[0])):
                    if i == 0:
                        parent_path = f"{pathlib.Path(file_dialog[0][i]).parents[0]}"
                        file_path = f"{pathlib.Path(file_dialog[0][i]).name}"
                        file_names += f"{parent_path} \n {file_path}"
                    else:
                        file_path = f"{pathlib.Path(file_dialog[0][i]).name}"
                        file_names += f"\n {file_path}"
                self.file_path.setText(file_names)
            else:
                path_to_file = pathlib.Path(file_dialog[0][0])
                self.file_path.setText(f"{str(path_to_file)}")

    def gen_channel_select(self):
        if self.is_fret.isChecked():
            self.input_layout.addWidget(self.fluorophore_channel_2)
            self.fluorophore_select_label.setText("Fluorophore channel names:")
        else:
            self.fluorophore_channel_2.setParent(
                None
            )  # so that widget does not exist anymore (.removeWidget only takes \
            # the widget out of the first parent, but keeps in any grandparents)
            self.fluorophore_select_label.setText("Fluorophore channel name:")


def gather_input():
    """

    launches the UI, gathers I/O

    :return: path_to_files & path_to_directory are pathlib.Paths, starvation_start & starvation_end are ints,
    acquisition_rate is float64 (imaging_rate), is_fret is bool (enables/disables selection of two fluorophore channels),
    channel_1 & channel_2 are strings (are not really channels, rather the data that is to be evaluated)

    """

    #### run the application ####
    app = QApplication(sys.argv)
    # sets style for path representation labels TODO: readable/pleasant on other systems?
    app.setStyleSheet("""QLabel#path {
                        border-style:outset;
                        border-color:#36393e;
                        border-width:4px;
                        background-color:#424549;
                        border-radius:10px;
                        color:white;
                        }""")
    window = MainWindow()
    window.end.clicked.connect(
        app.closeAllWindows
    )  # OK button gets connected to closing the application
    window.show()
    app.exec()
    #### run the application ####
    # after the application ends, collect values:
    #### I/O ####
    path_to_files = window.selected_files
    starvation_start = int(window.starvation_start_frame.text())
    starvation_end = int(window.starvation_end_frame.text())
    channel_1 = window.fluorophore_channel_1.currentText()
    channel_2 = window.fluorophore_channel_2.text()
    slope_index = int(window.slope_index.text())
    slope_multiplier = float(window.slope_multiplier.text())

    # tester print
    # all_values=(path_to_files, path_to_directory, starvation_start, starvation_end, acquisition_rate, is_fret, \
    # channel_1, channel_2)
    # for value in all_values:
    #     print(value)
    #     print(type(value))

    return (
        path_to_files,
        starvation_start,
        starvation_end,
        channel_1,
        channel_2,
        slope_index,
        slope_multiplier,
    )
    # is_starvation
    #### I/O ####


def setup_directory(saving_directory, name_of_file):
    """
    creates directories necessary for programmes output
    :param path_name: path to directory where the output is to be saved + the name of the original experiment
    :return: pathlib.Paths of each directory
    """

    concat_path_name = saving_directory / name_of_file
    figures_path = concat_path_name / "Figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    single_csvs_path = concat_path_name / "Single_Cell_CSVs"
    single_csvs_path.mkdir(parents=True, exist_ok=True)
    return figures_path, single_csvs_path


if __name__ == "__main__":
    gather_input()
