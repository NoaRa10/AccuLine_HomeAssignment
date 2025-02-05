# Create a csv file with the samples' number and label, then add the down-sampled signal to match 2017 data

import pandas as pd
import numpy as np
from ecg_processing_helper_functions import add_ecg_data_to_csv, resample_ecg_signal

# define the required paths
noise_path = "AccuLine_HomeAssignment/training2011/RECORDS-unacceptable.txt"
clean_path = "AccuLine_HomeAssignment/training2011/RECORDS-acceptable.txt"
output_csv_path = "AccuLine_HomeAssignment/2011_data.csv"

# Load the data from the text files
noise_data = np.loadtxt(noise_path, dtype=str)
clean_data = np.loadtxt(clean_path, dtype=str)

# Create DataFrames for each with tags
noise_df = pd.DataFrame({"record_name": noise_data, "label": "N"})
clean_df = pd.DataFrame({"record_name": clean_data, "label": "C"})

# Combine the dataframes
data_file_concat = pd.concat([noise_df, clean_df], ignore_index=True)

# Sort the combined data by the 'value' column
data_file_sorted = data_file_concat.sort_values(by="record_name").reset_index(drop=True)

# Save to CSV
data_file_sorted.to_csv(output_csv_path, index=False)

print(f"Sorted CSV saved to {output_csv_path}")

ecg_folder = "AccuLine_HomeAssignment/training2011"
data_file = add_ecg_data_to_csv(data_file_sorted, ecg_folder, file_type=".txt", fs=500)

# Resample the signal
data_file.rename(columns={data_file.columns[-1]: "ECG_Signal_original"}, inplace=True)
data_file["ECG_Signal"] = data_file["ECG_Signal_original"].apply(lambda x: resample_ecg_signal(x, original_fs=500, new_fs=300))

# Save the updated CSV
data_file.to_csv(output_csv_path, index=False)

