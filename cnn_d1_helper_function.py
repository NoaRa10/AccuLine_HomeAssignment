import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

def train(cnn1d_model, train_loader, optimizer, loss_function):

        cnn1d_model.train()  # Set model to training mode
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = cnn1d_model(x).squeeze()  # Squeeze to remove extra dimension

            # Compute the loss
            loss = loss_function(outputs, y.float())
            loss.backward()

            # Update weights
            optimizer.step()

            # Calculate accuracy
            predicted = (outputs >= 0.5).float()  # Convert probabilities to binary labels
            correct += (predicted == y).sum().item()
            total += y.size(0)

            train_loss += loss.item()

        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        return avg_train_loss, train_accuracy

def validate(cnn1d_model, valid_loader, loss_function):

    cnn1d_model.eval()  # Set model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in valid_loader:
            outputs = cnn1d_model(x).squeeze()
            loss = loss_function(outputs, y.float())

            val_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_val_loss = val_loss / len(valid_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


def evaluate_model(model, data_loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []  # For ROC AUC

    with torch.no_grad():  # Don't compute gradients for inference
        for inputs, labels in data_loader:

            outputs = model(inputs)

            # Predicted probabilities for ROC curve
            predicted_probs.extend(outputs.cpu().numpy())

            # Get the predicted labels (for binary classification, apply a threshold of 0.5)
            prediction = (outputs.squeeze() > 0.5).long()

            # Append predictions and true labels
            predicted_labels.extend(prediction.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    confusion = confusion_matrix(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_probs)

    return true_labels, predicted_labels, accuracy, precision, recall, f1, confusion, roc_auc, predicted_probs

def get_report_of_results(eval_type, accuracy, precision, recall, f1, confusion, roc_auc, true_labels, predicted_probs):

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Confusion Matrix:")
    print(confusion)

    print(f"{eval_type} ROC AUC Score: {roc_auc:.4f}")

    #Plot ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("ROC_Curve.png")
    plt.show()