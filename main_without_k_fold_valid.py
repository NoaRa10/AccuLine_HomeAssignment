from torch.utils.data import DataLoader
from my_dataset_class import MyDataset
from cnn1d_model_class import Cnn1dModel
from ecg_processing_helper_functions import *
from cnn_d1_helper_function_before_k_cross_valid import *
import torch.optim as optim
import torch.nn as nn

def main():

    save_path = "C:/Users/noara/AccuLine_HomeAssignment/" #"AccuLine_HomeAssignment/"
    data_path_2017 = f"{save_path}/2017_data_old.csv"
    data_path_2011 = f"{save_path}/2011_data_old.csv"

    concat_df = get_concat_data(data_path_2017, data_path_2011)

    # split the data into train and validation, do so noe, before segmenting the samples into equal 5sec segments
    # to prevent segments from the same sample to be divided to both training anf validation set.

    train_df, valid_df = split_data_to_train_and_validation(concat_df)

    # Now for each data set segment and balance the data set
    train_segmented_df = segment_ecg_signal_to_equal_length(train_df, sampling_freq=300, segment_size_seconds=5)
    train_balanced_df = balance_the_data(train_segmented_df, n_to_c_ratio=60/40)

    valid_segmented_df = segment_ecg_signal_to_equal_length(valid_df, sampling_freq=300, segment_size_seconds=5)
    valid_balanced_df = balance_the_data(valid_segmented_df, n_to_c_ratio=60/40)

    print(f"train portion of entire data {train_balanced_df.shape[0] / (train_balanced_df.shape[0] + valid_balanced_df.shape[0])}")
    print(f"validation portion of entire data {valid_balanced_df.shape[0] / (train_balanced_df.shape[0] + valid_balanced_df.shape[0])}")

    batch_size = 32

    train_data = MyDataset(train_balanced_df)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_data = MyDataset(valid_balanced_df)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    cnn1d_model = Cnn1dModel()
    # try adding weight_decay to the optimizer to improve generalization
    optimizer = optim.Adam(cnn1d_model.parameters(), lr=0.001)#, weight_decay=1e-4)
    # try adding scheduler to the optimizer
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_function = nn.BCELoss()

    # training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        avg_train_loss, train_accuracy = train(cnn1d_model, train_loader, optimizer, loss_function)#, scheduler)
        avg_val_loss, val_accuracy = validate(cnn1d_model, valid_loader, loss_function)

        # Print statistics
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # evaluate training and validation process
    (tr_true_labels, tr_predicted_labels, tr_accuracy, tr_precision, tr_recall, tr_f1, tr_confusion, tr_roc_auc,
     tr_predicted_probs) = evaluate_model(cnn1d_model, train_loader)
    print("Training Evaluation:")
    get_report_of_results("Training", tr_accuracy, tr_precision, tr_recall, tr_f1, tr_confusion, tr_roc_auc, tr_true_labels, tr_predicted_probs)

    (vl_true_labels, vl_predicted_labels, vl_accuracy, vl_precision, vl_recall, vl_f1, vl_confusion, vl_roc_auc,
     vl_predicted_probs) = evaluate_model(cnn1d_model, valid_loader)
    print("Validation Evaluation:")
    get_report_of_results("Validation", vl_accuracy, vl_precision, vl_recall, vl_f1, vl_confusion, vl_roc_auc, vl_true_labels,
                          vl_predicted_probs)

    # test model performance
    test_path = f"{save_path}/2017_test_data.csv"
    test_df = pd.read_csv(test_path, usecols=["label", "sample"])
    test_df["sample"] = test_df["sample"].apply(lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=','))
    test_segmented_df = segment_ecg_signal_to_equal_length(test_df, sampling_freq=300, segment_size_seconds=5)
    test_balanced_df = balance_the_data(test_segmented_df, n_to_c_ratio=60/40)

    test_data = MyDataset(test_balanced_df)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    (ts_true_labels, ts_predicted_labels, ts_accuracy, ts_precision, ts_recall, ts_f1, ts_confusion, ts_roc_auc,
     ts_predicted_probs) = evaluate_model(cnn1d_model, test_loader)
    print("Test Evaluation:")
    get_report_of_results("Test", ts_accuracy, ts_precision, ts_recall, ts_f1, ts_confusion, ts_roc_auc, ts_true_labels,
                          ts_predicted_probs)

    # Add the predictions as a new column to the test DataFrame
    test_balanced_df['predicted_label'] = ts_predicted_labels
    test_balanced_df.to_csv(f"{save_path}/2017_test_data_including_prediction.csv", index=False)

if __name__ == "__main__":
    main()
