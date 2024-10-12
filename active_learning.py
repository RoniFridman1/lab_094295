import copy
import time
import os
import json
from model import train_model
import torch
from Config import Config
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from select_samples_methods import select_samples


def evaluate_model(model, data_loader, output_dir='output', iteration=None, print_metrics=False):
    """
    Evaluates the model on a given dataset with additional metrics and saves results to files.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        output_dir (str): Directory to save output files.
        iteration (int, optional): Iteration number for saving files with unique names.
        print_metrics (boolean): If to print the metrics or not

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
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    if print_metrics:
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
        "confusion_matrix": conf_matrix.tolist()}  # Convert to list for JSON serialization

    metrics_filename = f"metrics_iteration_{iteration}.json" if iteration is not None else "metrics.json"
    with open(os.path.join(output_dir, metrics_filename), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix as an image
    ConfusionMatrixDisplay(conf_matrix).plot()
    conf_matrix_filename = f"confusion_matrix_iteration_{iteration}.png" if iteration is not None else "confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, conf_matrix_filename))
    plt.close()  # Close the plot to avoid display overlap in loops

    return metrics


def active_learning_loop(model, train_generator, val_generator, test_generator,
                         unlabeled_data, method, config, output_dir='output'):
    """
    Main loop for Active Learning.

    Args:
        model (torch.nn.Module): The machine learning model to be trained.
        train_generator (DataLoader): DataLoader for training data.
        val_generator (DataLoader): DataLoader for validation data.
        test_generator (DataLoader): DataLoader for test data.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        method (str): Strategy for sample selection (e.g. 'uncertainty', 'entropy', 'random').
        config (Config): Experiment configuration.
        output_dir (str): a directory path where to store the evaluation metrics.

    Returns:
        model (torch.nn.Module): The trained model after Active Learning.
    """
    metrics = []
    for j in range(config.ACTIVE_LEARNING_ITERATIONS):
        t0 = time.time()
        print(
            f"Active Learning Iteration {j + 1}/{config.ACTIVE_LEARNING_ITERATIONS}."
            f"\tTrain Samples: {(j+1) * config.SAMPLES_PER_ITERATION}"
            f"\tUnlabeled: {config.TOTAL_TRAINING_SAMPLES - (j+1) * config.SAMPLES_PER_ITERATION}")
        iter_model = copy.deepcopy(model)  # We want to start the model from scratch for every iteration.
        if len(unlabeled_data) <= 0:
            break
        # Train the model on current labeled data
        iter_model = train_model(iter_model, train_generator, val_generator, epochs=config.MODEL_TRAINING_EPOCHS,
                                 learning_rate=config.leaning_rate)

        # Select new samples to be labeled
        selected_samples, selected_labels = select_samples(iter_model, unlabeled_data, config=config, strategy=method,
                                                           num_samples=config.SAMPLES_PER_ITERATION)

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
        metrics.append(evaluate_model(iter_model, test_generator, iteration=j, output_dir=output_dir))
        print(f"Time of Iteration: {round(time.time()-t0)} sec")
    return metrics
