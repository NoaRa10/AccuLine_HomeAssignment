from scipy.io import loadmat
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import numpy as np
from data_exploration_helper_function import *

def save_2017_data(csv_path, mat_folder, output_csv_path):
    # Load the CSV file abd make a copy
    data_file_orig = pd.read_csv(csv_path, header=None, names=["record_name", "label"])  # assign custom column names
    data_file = data_file_orig.copy()

    # Since the task is noise recognition I currently do not care for the different rhythm identified.
    # Modify labels: '~' → 'N', everything else → 'C'
    data_file["label"] = data_file["label"].apply(lambda x: "N" if x == "~" else "C")

    data_file = add_ecg_data_to_csv(data_file, mat_folder, file_type=".mat", fs=300)

    # Save the updated CSV
    data_file.to_csv(output_csv_path, index=False)
    return data_file

def add_ecg_data_to_csv(data_file, ecg_folder, file_type, fs):
    """Adds ECG data from .mat file or .txt to a DataFrame and saves it as a CSV.
    Parameters:
    - data_file: Pandas DataFrame containing the CSV with labels and filenames.
    - ecg_folder: Path to the folder containing the .mat or .txt files.
    """
    # Add a new column for the extracted ECG data
    data_file["sample"] = None
    acceptable_min_duration_sec = 5
    acceptable_length = acceptable_min_duration_sec * fs

    # Process each row
    for index, row in data_file.iterrows():
        sample_name = row["record_name"]
        file_path = f"{ecg_folder}/{sample_name}{file_type}"
        current_signal = []

        if file_type == ".mat":
            try:
                mat_data = loadmat(file_path)
                current_signal = mat_data['val']
                current_signal = current_signal.reshape(len(current_signal[0]), )
            except Exception as e:
                print(f"Error loading {sample_name}: {e}")

        elif file_type == ".txt":
            try:
                data = pd.read_csv(file_path, header=None)
                data_without_index = data.iloc[:, 1:] # Remove the first column (index column)
                current_signal = data_without_index.values.flatten() # flatten the DataFrame

            except Exception as e:
                print(f"Error loading {sample_name}: {e}")

        #current_signal = current_signal.tolist()
        if len(current_signal) >= acceptable_length:
            data_file.at[index, "sample"] = current_signal#.tolist()  # Convert to list for CSV compatibility

    return data_file

def resample_ecg_signal(ecg_signal, original_fs, target_fs):
    """Resamples the given ECG signal from original_fs to new_fs.
    Parameters:
    - sample: signal to resample.
    - original_fs: original sampling frequency in Hz.
    - target_fs: desired sampling frequency in Hz.
    """
    try:
        resampled_signal = signal.resample_poly(ecg_signal, target_fs, original_fs)
        return resampled_signal

    except Exception as e:
        print(f"Error resampling signal: {e}")
        return None

def get_concat_data(input1, input2, input_type):
    """Combine two data frames: 2017 and 2011 data
    Parameters:
    - input1: path for 2017 date or data frame for 2011
    - input2: path for 2011 date or data frame for 2017
    - input_type: "path" or "data" - specify how to combine data frames - using paths or dataframes
    """

    if input_type == "path":
        # Read data frames and change signal format to np array
        df1 = pd.read_csv(input1, usecols=["label", "sample"])
        df1["sample"] = df1["sample"].apply(lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=','))
        df2 = pd.read_csv(input2, usecols=["label", "sample"])
        df2["sample"] = df2["sample"].apply(lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=' '))
    elif input_type == "data":
        df1 = input1[["sample", "label"]]
        df2 = input2[["sample", "label"]]

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Add sample id for proper training-validation split after segmentation
    df['sample_id'] = df.index

    return df

def segment_ecg_signal_to_equal_length(df, sampling_freq=300, segment_size_seconds=5):
    """ Segment the ecg signal to equal length for training anf for later application
        Parameters:
    - df: full data frame
    - sampling_freq: sampling frequency in Hz to define length for segmentation
    - segment_size_seconds: required segment time in sec to define length for segmentation
    """
    segment_length = segment_size_seconds * sampling_freq

    segments = []

    for index, row in df.iterrows():
        current_signal = row["sample"]
        current_label = row["label"]
        current_id = row["sample_id"]
        current_label = 0 if current_label == 'N' else 1  # Convert 'N' to 0 (noisy), 'C' to 1 (clean)

        for i in range(0, len(current_signal), segment_length):
            segment = current_signal[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append({"sample": segment, "label": current_label, "sample_id": current_id})

    segmented_df = pd.DataFrame(segments)
    return segmented_df

def balance_the_data(df, n_to_c_ratio):
    """Data in highly unbalanced between the two labels, get more balanced data
    """
    label_counts = get_labels_ration(df)
    min_label = label_counts.idxmin()
    max_label = label_counts.idxmax()

    min_count = label_counts[min_label]
    max_count = int(min_count * n_to_c_ratio)

    df_min = df[df["label"] == min_label]  # Take all from the smaller class
    df_max = df[df["label"] == max_label].sample(n=max_count, random_state=42)  # Sample from the larger class

    balanced_df = pd.concat([df_min, df_max]).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def split_data_to_train_and_validation(df):
    """Prepare the data frames for training and validation by shuffling the data and divide
        Parameters:
        df: full data frame of segmented ECG signal
        !!!! Was in use in previous version, no longer used.
    """
    # Shuffle the dataset
    shuffle_df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # random_state for reproducibility

    # Split into training (80%) and validation (20%)
    train_df, valid_df = train_test_split(shuffle_df, test_size=0.2, random_state=42)

    # Reset indices after shuffle
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {valid_df.shape}")

    return train_df, valid_df
