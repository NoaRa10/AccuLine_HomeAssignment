# Creat a copy of the .csv label file and add the time series data to it
from ecg_processing_helper_functions import *

# define the required paths
csv_path = "AccuLine_HomeAssignment/test/REFERENCE.csv"
mat_folder = "AccuLine_HomeAssignment/test"
output_csv_path = "AccuLine_HomeAssignment/2017_test_data.csv"

save_2017_data(csv_path, mat_folder, output_csv_path)