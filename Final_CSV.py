import polars as pl
from pathlib import Path
import numpy as np

from polars.exceptions import NoDataError


def Final_CSV(concat_path_name, path_for_pairs):

    # Files with the min-max indeces in the columns
    all_files = Path(path_for_pairs).glob('*.csv')
    csvs = None ### Still get UnboundLocalError: local variable 'csvs' referenced before assignment for the Whi8 KO data
    #Concatenate all csv with index values for Whi5 min and max
    for f in all_files:
        try:
            csvs = [pl.read_csv(f) for f in all_files]
        except NoDataError:
            pass
    df_index = pl.concat(csvs)

    df_sorted = df_index.sort(['Cell_ID','Maxima_Index'])
    df_sorted.write_csv(concat_path_name / 'All.csv', separator = ",") # index=False gives same results as line 14 (reset index)
