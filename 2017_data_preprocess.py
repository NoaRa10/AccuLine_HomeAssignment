# Creat a copy of the .csv label file and add the time series data to it
from ecg_processing_helper_functions import *

# You can run this function for the train-valid set of for the test set
data_type = 'test' #'train-valid'

# define the required paths
if data_type == 'train-valid':
    csv_path = "C:/Users/noara/AccuLine_HomeAssignment/training2017/REFERENCE-original.csv"
    mat_folder = "C:/Users/noara/AccuLine_HomeAssignment/training2017"
    output_csv_path = "C:/Users/noara/AccuLine_HomeAssignment/2017_data.csv"
elif data_type == 'test':
    csv_path = "C:/Users/noara/AccuLine_HomeAssignment/test/REFERENCE.csv"
    mat_folder = "C:/Users/noara/AccuLine_HomeAssignment/test"
    output_csv_path = "C:/Users/noara/AccuLine_HomeAssignment/2017_test_data.csv"

save_2017_data(csv_path, mat_folder, output_csv_path)
