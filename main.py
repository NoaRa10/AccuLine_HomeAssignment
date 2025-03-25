from torch.utils.data import DataLoader, Dataset, Subset
from my_dataset_class import MyDataset
from cnn1d_model_class import Cnn1dModel
from ecg_processing_helper_functions import *
from cnn_d1_helper_function import *
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():

    save_path = "C:/Users/noara/AccuLine_HomeAssignment/" #"AccuLine_HomeAssignment/"
    data_path_2017 = f"{save_path}/2017_data.csv"
    data_path_2011 = f"{save_path}/2011_data.csv"

    concat_df = get_concat_data(data_path_2017, data_path_2011)
    segmented_df = segment_ecg_signal_to_equal_length(concat_df, sampling_freq=300, segment_size_seconds=5)

    # Define batch size and group ratio
    batch_size = 32
    n_to_c_ratio = 60/40

    # Define K-Folds
    k_folds = 7
    group_k_fold = GroupKFold(n_splits=k_folds)
    sample_id = segmented_df['sample_id'].values

    # Track model's performance
    fold_results = {
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "valid_accuracy": [],
        "valid_precision": [],
        "valid_recall": [],
        "valid_f1": []
    }

    for fold, (train_idx, valid_idx) in enumerate(group_k_fold.split(segmented_df, groups=sample_id)):
        print(f"Fold {fold}: Train size = {len(train_idx)}, Validation size = {len(valid_idx)}")

        # Get the train and validation DataFrames for this fold
        train_df = segmented_df.iloc[train_idx]
        valid_df = segmented_df.iloc[valid_idx]

        # Apply data balancing function
        train_balanced_df = balance_the_data(train_df, n_to_c_ratio=n_to_c_ratio)
        valid_balanced_df = balance_the_data(valid_df, n_to_c_ratio=n_to_c_ratio)

        # Create the train and validation datasets
        train_dataset = MyDataset(train_balanced_df)
        valid_dataset = MyDataset(valid_balanced_df)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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
                f"Fold {fold} - Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        #Evaluate the model after training
        (tr_true_labels, tr_pred_labels, tr_accuracy, tr_precision, tr_recall, tr_f1, tr_confusion, tr_roc_auc,
         _) = evaluate_model(cnn1d_model, train_loader)
        (vl_true_labels, vl_pred_labels, vl_accuracy, vl_precision, vl_recall, vl_f1, vl_confusion, vl_roc_auc,
         _) = evaluate_model(cnn1d_model, valid_loader)

        print(
            f"Fold {fold} - Training Metrics: Accuracy: {tr_accuracy:.2f}%, F1: {tr_f1:.4f}, ROC AUC: {tr_roc_auc:.4f}")
        print(
            f"Fold {fold} - Validation Metrics: Accuracy: {vl_accuracy:.2f}%, F1: {vl_f1:.4f}, ROC AUC: {vl_roc_auc:.4f}")

        #Store results
        fold_results["train_accuracy"].append(tr_accuracy)
        fold_results["train_precision"].append(tr_precision)
        fold_results["train_recall"].append(tr_recall)
        fold_results["train_f1"].append(tr_f1)
        fold_results["valid_accuracy"].append(vl_accuracy)
        fold_results["valid_precision"].append(vl_precision)
        fold_results["valid_recall"].append(vl_recall)
        fold_results["valid_f1"].append(vl_f1)

    # Summarize cross-validation results
    print("\n=== Cross-Validation Summary ===")
    print(
        f"Train Accuracy: {np.mean(fold_results['train_accuracy']):.4f} ± {np.std(fold_results['train_accuracy']):.4f}")
    print(f"Train Precision: {np.mean(fold_results['train_precision']):.4f} ± {np.std(fold_results['train_precision']):.4f}")
    print(f"Train Recall: {np.mean(fold_results['train_recall']):.4f} ± {np.std(fold_results['train_recall']):.4f}")
    print(f"Train F1 Score: {np.mean(fold_results['train_f1']):.4f} ± {np.std(fold_results['train_f1']):.4f}")
    print(
        f"Validation Accuracy: {np.mean(fold_results['valid_accuracy']):.4f} ± {np.std(fold_results['valid_accuracy']):.4f}")
    print(f"Validation Precision: {np.mean(fold_results['valid_precision']):.4f} ± {np.std(fold_results['valid_precision']):.4f}")
    print(f"Validation Recall: {np.mean(fold_results['valid_recall']):.4f} ± {np.std(fold_results['valid_recall']):.4f}")
    print(f"Validation F1 Score: {np.mean(fold_results['valid_f1']):.4f} ± {np.std(fold_results['valid_f1']):.4f}")

    # test model performance
    test_path = f"{save_path}/2017_test_data.csv"
    test_df = pd.read_csv(test_path, usecols=["label", "sample"])
    test_df["sample"] = test_df["sample"].apply(lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=','))
    test_df['sample_id'] = test_df.index
    test_segmented_df = segment_ecg_signal_to_equal_length(test_df, sampling_freq=300, segment_size_seconds=5)
    test_balanced_df = balance_the_data(test_segmented_df, n_to_c_ratio=n_to_c_ratio)

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
