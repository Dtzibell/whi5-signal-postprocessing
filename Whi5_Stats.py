# Take output of Nuclear_Signal_Loop/SingleCell and calculate G1 to full Whi5 exit duration, 50% out from peak duration (textbook START pass)

import pathlib
import pandas as pd
import numpy as np
from File_Importer import File_Import

# Import as usual with File_Importer and create a subfolder in Results/Exp_Name/ called Whi5_Analysis
# Call function File_Importer to open file to process, rs = full path, file_name = base name with extension
rs, file_name = File_Import()

# Get file name wihout extension
file_name = (file_name[:-4])

# Make directory "Processed_Results" on Desktop, with subdirectories named after file loaded FIX FOR FOLDER NAME, base file name is always All_upto_No.csv
# Ideally put in the same folder as the Nuclear Signal results
results_path_full = str('/Users/dreadwolf/Desktop/Processed_Results/' + file_name)
results_path = pathlib.Path(results_path_full)
results_path.mkdir(parents = True, exist_ok=True)
results_path_name = str(results_path)

# Final path to save results as csv from Array_Pairs
path_name = '/Users/dreadwolf/Desktop/Processed_Results/' + file_name + '/'

#Ask for imaging frame rate / Unecessary, calculations done on time (min)
#img_interval = int(input("Enter imaging rate (seconds):"))

# Read CSV file into DataFrame df
df = pd.read_csv(rs)
line_N = len(df)    

whi5_peak_to_exit = np.zeros((line_N, 3))
whi5_peak_to_exit[:,2] = df['Cell_ID']

# Duration in min from G1 peak to full nuclear exit
whi5_peak_to_exit[:,0] = (df['Minima_Time_(min)'] - df['Maxima_Time_(min)'])
# Duration in min from G1 peak to START (50% Whi5 out)
whi5_peak_to_exit[:,1] = (((df['Minima_Time_(min)'] + df['Maxima_Time_(min)'])/2) - df['Maxima_Time_(min)'])    ###* (img_interval/60)  

# Storage
whi5_analysis_store = pd.DataFrame(whi5_peak_to_exit)
whi5_analysis_store.columns = ['G1 Peak to Full Exit (min)', 'START Duration (min)', 'Cell ID']
whi5_analysis_store.to_csv(path_name + 'Whi5_Analysed.csv')