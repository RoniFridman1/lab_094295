import os, json
from model import train_model
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def select_samples(model, unlabeled_data, strategy='uncertainty', num_samples=10):
    """
    Selects the most informative samples based on the chosen strategy.

    Args:
        model (torch.nn.Module): Trained model used to select samples.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        strategy (str): Strategy for selecting samples ('uncertainty', 'entropy', 'random').
        num_samples (int): Number of samples to select.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    indices = []
    all_labels = []

    # Compute scores for each sample in the unlabeled data
    with torch.no_grad():
        for i, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            outputs = model(images)  # Get raw logits
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities

            # Store the labels for later use
            all_labels.extend(labels.numpy())

            # Uncertainty sampling: select samples with probabilities closest to 0.5
            if strategy == 'uncertainty':
                uncertainty_scores = 1 - torch.max(probabilities, dim=1).values  # 1 - max probability
                scores.extend(uncertainty_scores.cpu().numpy())
                indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])

            # Entropy sampling: select samples with highest entropy
            elif strategy == 'entropy':
                entropy_scores = -torch.sum(probabilities * torch.log(probabilities + 1e-10),
                                            dim=1)  # Add epsilon to avoid log(0)
                scores.extend(entropy_scores.cpu().numpy())
                indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])

            # Random sampling: select random samples (no scores needed)
            elif strategy == 'random':
                indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])

    # Convert scores to numpy array for sorting
    if strategy in ['uncertainty', 'entropy']:
        scores = np.array(scores)
        indices = np.array(indices)
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        selected_indices = sorted_indices[:num_samples]  # Select top samples
        selected_samples = indices[selected_indices]  # Get corresponding sample indices
    else:
        # For random strategy, select random indices
        selected_samples = np.random.choice(indices, size=num_samples, replace=False)

    selected_labels = [all_labels[idx] for idx in selected_samples]

    return selected_samples.tolist(), selected_labels


def active_learning_loop(model, train_generator, val_generator, test_generator, unlabeled_data, method, iterations=5,
                         samples_per_iteration=10, model_train_epochs=2, output_dir='output'):
    """
    Main loop for Active Learning.

    Args:
        model (torch.nn.Module): The machine learning model to be trained.
        train_generator (DataLoader): DataLoader for training data.
        val_generator (DataLoader): DataLoader for validation data.
        test_generator (DataLoader): DataLoader for test data.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        method (str): Strategy for sample selection (e.g., 'uncertainty', 'entropy', 'random').
        iterations (int): Number of Active Learning iterations.
        samples_per_iteration (int): Number of samples to query per iteration.

    Returns:
        model (torch.nn.Module): The trained model after Active Learning.
    """
    metrics = []
    for j in range(iterations):
        print(
            f"Active Learning Iteration {j + 1}/{iterations}.\tTrain Samples: {len(train_generator) * train_generator.batch_size}"
                                +f"\tUnlabeled: {len(unlabeled_data)* unlabeled_data.batch_size}")
        if len(unlabeled_data) <= 0:
            break
        # Train the model on current labeled data
        train_model(model, train_generator, val_generator, epochs=model_train_epochs)

        # Select new samples to be labeled
        selected_samples, selected_labels = select_samples(model, unlabeled_data, strategy=method,
                                                           num_samples=samples_per_iteration)

        # Retrieve selected images and labels from the unlabeled dataloader
        train_generator.dataset.indices = train_generator.dataset.indices + [unlabeled_data.dataset.indices[i]
                                                                             for i in
                                                                             range(len(unlabeled_data.dataset.indices))
                                                                             if i in selected_samples]
        updated_train_data = train_generator.dataset

        # Add new labeled samples to the existing training data
        unlabeled_data.dataset.indices = [unlabeled_data.dataset.indices[i]
                                          for i in range(len(unlabeled_data.dataset.indices))
                                          if i not in selected_samples]
        updated_unlabeled_data = unlabeled_data.dataset

        # Recreate train dataloader with updated data
        train_generator = DataLoader(updated_train_data, batch_size=train_generator.batch_size, shuffle=True)
        unlabeled_data = DataLoader(updated_unlabeled_data, batch_size=unlabeled_data.batch_size, shuffle=False)

        # Evaluate the model
        metrics.append(evaluate_model(model, test_generator, iteration=j, output_dir=output_dir))
    return model,metrics


def evaluate_model(model, data_loader, output_dir='output', iteration=None):
    """
    Evaluates the model on a given dataset with additional metrics and saves results to files.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        output_dir (str): Directory to save output files.
        iteration (int, optional): Iteration number for saving files with unique names.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    # Save metrics to a JSON file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
    }
    metrics_filename = f"metrics_iteration_{iteration}.json" if iteration is not None else "metrics.json"
    with open(os.path.join(output_dir, metrics_filename), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix as an image
    ConfusionMatrixDisplay(conf_matrix).plot()
    conf_matrix_filename = f"confusion_matrix_iteration_{iteration}.png" if iteration is not None else "confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, conf_matrix_filename))
    plt.close()  # Close the plot to avoid display overlap in loops

    return metrics
