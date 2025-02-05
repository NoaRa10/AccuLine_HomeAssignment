from ecg_processing_helper_functions import *

def main():

    save_path = "AccuLine_HomeAssignment/"
    data_path_2017 = f"{save_path}/2017_data.csv"
    data_path_2011 = f"{save_path}/2011_data.csv"

    concat_df = get_concat_data(data_path_2017, data_path_2011)
    get_ecg_signal_lengths(concat_df)
    get_labels_ration(concat_df)

    # lengths per each label
    clean_df = concat_df[concat_df['label'] == 'C']
    get_ecg_signal_lengths(clean_df, get_fig=True)
    mean_len_clean, std_len_clean = get_ecg_signal_mean_std(clean_df)
    print(f"Clean signal mean length = {mean_len_clean}, std = {std_len_clean}")
    noisy_df = concat_df[concat_df['label'] == 'N']
    get_ecg_signal_lengths(noisy_df, get_fig=True)
    mean_len_noisy, std_len_noisy = get_ecg_signal_mean_std(noisy_df)
    print(f"Noisy signal mean length = {mean_len_noisy}, std = {std_len_noisy}")

if __name__ == "__main__":
    main()