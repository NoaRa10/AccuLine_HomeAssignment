import matplotlib.pyplot as plt
import numpy as np

def ecg_visualization(signal, sampling_rate, num_point_to_plot=None, plot_on_existing_fig: bool = False):

    if not sampling_rate:
        print("sampling rate not provided")
        return

    if not num_point_to_plot:
        num_point_to_plot = len(signal)

    ecg_signal = signal[:num_point_to_plot]  # This is just random data as an example

    # Create a time axis (assuming a sampling rate of 300Hz for this example)
    time = np.arange(0, len(ecg_signal)) / sampling_rate  # Time vector for the signal

    # Plot the signal
    if not plot_on_existing_fig:
        plt.figure(figsize=(10, 6))  # Set the figure size

    plt.plot(time, ecg_signal, label='ECG Signal')  # Plot the signal with a label
    plt.xlabel('Time (seconds)')  # Label for the x-axis
    plt.ylabel('Amplitude')  # Label for the y-axis
    plt.title('ECG Signal')  # Title of the plot
    plt.legend()  # Show the legend
    plt.grid(True)  # Show grid lines for better readability
    plt.savefig("ecg_signal_example.png")
    plt.show()  # Display the plot