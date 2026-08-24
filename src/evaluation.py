"""
evaluation.py

Evaluation utilities for the Driver Drowsiness Detection models:
training curve visualization, confusion matrices, and classification
reports for comparing the custom CNN and MobileNetV2 models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# 1. Training curves
def plot_training_curves(custom_cnn_history, mobilenetv2_history):
    """
    Plot training/validation accuracy and loss curves for both models,
    side by side, to visually compare convergence and overfitting
    behavior across epochs.

    Parameters
    ----------
    custom_cnn_history : keras.callbacks.History
        Object returned by custom_cnn_model.fit(...).
    mobilenetv2_history : keras.callbacks.History
        Object returned by mobilenetv2_model.fit(...).

    Returns
    -------
    None. Displays the plot via plt.show().
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(custom_cnn_history.history['accuracy'], label='Train', marker='o')
    axes[0, 0].plot(custom_cnn_history.history['val_accuracy'], label='Validation', marker='o')
    axes[0, 0].set_title('Custom CNN - Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    axes[0, 1].plot(custom_cnn_history.history['loss'], label='Train', marker='o')
    axes[0, 1].plot(custom_cnn_history.history['val_loss'], label='Validation', marker='o')
    axes[0, 1].set_title('Custom CNN - Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    axes[1, 0].plot(mobilenetv2_history.history['accuracy'], label='Train', marker='o', color='green')
    axes[1, 0].plot(mobilenetv2_history.history['val_accuracy'], label='Validation', marker='o', color='orange')
    axes[1, 0].set_title('MobileNetV2 - Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()

    axes[1, 1].plot(mobilenetv2_history.history['loss'], label='Train', marker='o', color='green')
    axes[1, 1].plot(mobilenetv2_history.history['val_loss'], label='Validation', marker='o', color='orange')
    axes[1, 1].set_title('MobileNetV2 - Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


# 2. Confusion matrices
def plot_confusion_matrices(true_labels, custom_cnn_predictions, mobilenetv2_predictions):
    """
    Plot side-by-side confusion matrix heatmaps for the custom CNN and
    MobileNetV2 predictions on the same set of true labels (typically
    the held-out test set).

    Parameters
    ----------
    true_labels : array-like
        Ground-truth binary labels (0 = Non Drowsy, 1 = Drowsy).
    custom_cnn_predictions : array-like
        Binary predictions (0/1) from the custom CNN model.
    mobilenetv2_predictions : array-like
        Binary predictions (0/1) from the MobileNetV2 model.

    Returns
    -------
    None. Displays the plot via plt.show().
    """
    custom_cnn_cm = confusion_matrix(true_labels, custom_cnn_predictions)
    mobilenetv2_cm = confusion_matrix(true_labels, mobilenetv2_predictions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(custom_cnn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Non Drowsy', 'Drowsy'], yticklabels=['Non Drowsy', 'Drowsy'])
    axes[0].set_title('Custom CNN - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(mobilenetv2_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Non Drowsy', 'Drowsy'], yticklabels=['Non Drowsy', 'Drowsy'])
    axes[1].set_title('MobileNetV2 - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


# 3. Classification reports
def print_classification_reports(true_labels, custom_cnn_predictions, mobilenetv2_predictions):
    """
    Print sklearn classification reports (precision, recall, F1-score
    per class) for both models on the same set of true labels.

    Parameters
    ----------
    true_labels : array-like
        Ground-truth binary labels (0 = Non Drowsy, 1 = Drowsy).
    custom_cnn_predictions : array-like
        Binary predictions (0/1) from the custom CNN model.
    mobilenetv2_predictions : array-like
        Binary predictions (0/1) from the MobileNetV2 model.

    Returns
    -------
    None. Prints both reports to stdout.
    """
    print("Custom CNN - Classification Report:")
    print(classification_report(true_labels, custom_cnn_predictions, target_names=['Non Drowsy', 'Drowsy']))

    print("\nMobileNetV2 - Classification Report:")
    print(classification_report(true_labels, mobilenetv2_predictions, target_names=['Non Drowsy', 'Drowsy']))