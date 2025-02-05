import matplotlib.pyplot as plt
def get_ecg_signal_lengths(df, get_fig: bool=False):

    signal_lengths = df['ECG_Signal'].apply(len)
    min_length = signal_lengths.min()
    max_length = signal_lengths.max()

    print(f"Min length: {min_length}, Max length: {max_length}")

    if get_fig:
        plt.hist(signal_lengths, bins=10, edgecolor='black')
        plt.xlabel('Signal Length')
        plt.ylabel('Count')
        plt.title('Histogram of ECG Signal Lengths')
        plt.savefig("ecg_histogram.png")
        plt.show()

def get_ecg_signal_mean_std(df):
    signal_lengths = df['ECG_Signal'].apply(len)
    mean_length = signal_lengths.mean()
    std_length = signal_lengths.std()

    return mean_length, std_length

def get_labels_ration(df):
    label_counts = df['label'].value_counts()
    print(label_counts)
    return label_counts


