from model import train_model
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def select_samples(model, unlabeled_data, strategy='uncertainty', num_samples=10):
    """
    Placeholder function for selecting the most informative samples.

    Args:
        model (tf.keras.Model): Trained model used to select samples.
        unlabeled_data (np.array): Unlabeled data pool.
        strategy (str): Strategy for selecting samples (e.g., 'uncertainty', 'entropy').
        num_samples (int): Number of samples to select.

    Returns:
        selected_samples (np.array): Indices of selected samples.
    """
    # Implement specific strategies here
    # Placeholder code using random selection:
    selected_samples = np.random.choice(len(unlabeled_data), size=num_samples, replace=False)

    return selected_samples


def active_learning_loop(model, train_generator, val_generator, test_generator, unlabeled_data, iterations=5,
                         samples_per_iteration=10):
    """
    Main loop for Active Learning.

    Args:
        model (tf.keras.Model): The machine learning model to be trained.
        train_generator, val_generator, test_generator: Data generators.
        unlabeled_data (np.array): Pool of unlabeled data.
        iterations (int): Number of Active Learning iterations.
        samples_per_iteration (int): Number of samples to query per iteration.

    Returns:
        model (tf.keras.Model): The trained model after Active Learning.
    """
    for i in range(iterations):
        print(f"Active Learning Iteration {i + 1}/{iterations}")

        # Train the model on current labeled data
        train_model(model, train_generator, val_generator, epochs=5)

        # Select new samples to be labeled
        selected_samples = select_samples(model, unlabeled_data, num_samples=samples_per_iteration)

        # Simulate labeling process
        # TODO: Replace with actual labeling function or human annotation
        print(f"Selected samples for labeling: {selected_samples}")

        # Update training data with newly labeled samples
        # TODO: Implement data update logic

        # Evaluate the model
        evaluate_model(model, test_generator)

    return model


def evaluate_model(model, data_loader):
    """
    Evaluates the model on a given dataset with additional metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Display metrics
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    # Display confusion matrix
    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix
    }
