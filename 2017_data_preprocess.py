# Creat a copy of the .csv label file and add the time series data to it
from ecg_processing_helper_functions import *

# define the required paths
csv_path = "AccuLine_HomeAssignment/training2017/REFERENCE-original.csv"
mat_folder = "AccuLine_HomeAssignment/training2017"
output_csv_path = "AccuLine_HomeAssignment/2017_data.csv"

save_2017_data(csv_path, mat_folder, output_csv_path)
