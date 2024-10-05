import copy
import time
import os
import json
from tqdm import tqdm

import Config
from model import train_model
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_loader import load_data
from Config import Config
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

def _select_samples(model, unlabeled_data, config, strategy='uncertainty', num_samples=10):
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
    if strategy == 'uncetainty':
        return uncertainty(model, unlabeled_data, num_samples)
    elif strategy == 'entropy':
        return entropy(model, unlabeled_data, num_samples)
    elif strategy == 'random':
        return random(model, unlabeled_data, num_samples, seed=config.seed)
    elif strategy == "pca_kmeans":
        return pca_kmeans(unlabeled_data, num_samples,seed=config.seed, pca_n_components=100)

def uncertainty(model, unlabeled_data,num_samples):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    indices = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            outputs = model(images)  # Get raw logits
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities

            # Store the labels for later use
            all_labels.extend(labels.numpy())

            entropy_scores = -torch.sum(probabilities * torch.log(probabilities + 1e-10),
                                        dim=1)  # Add epsilon to avoid log(0)
            scores.extend(entropy_scores.cpu().numpy())
            indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])
    scores = np.array(scores)
    indices = np.array(indices)
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    selected_indices = sorted_indices[:num_samples]  # Select top samples
    selected_samples = indices[selected_indices]  # Get corresponding sample indices
    selected_labels = [all_labels[idx] for idx in selected_samples]
    return selected_samples.tolist(), selected_labels

def entropy(model, unlabeled_data,num_samples):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    indices = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            outputs = model(images)  # Get raw logits
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities

            # Store the labels for later use
            all_labels.extend(labels.numpy())

            uncertainty_scores = 1 - torch.max(probabilities, dim=1).values  # 1 - max probability
            scores.extend(uncertainty_scores.cpu().numpy())
            indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])
    scores = np.array(scores)
    indices = np.array(indices)
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    selected_indices = sorted_indices[:num_samples]  # Select top samples
    selected_samples = indices[selected_indices]  # Get corresponding sample indices
    selected_labels = [all_labels[idx] for idx in selected_samples]
    return selected_samples.tolist(), selected_labels


def random(model, unlabeled_data,num_samples, seed=42):
    model.eval()
    indices = []
    all_labels = []
    for i, (images, labels) in enumerate(unlabeled_data):
        # Store the labels for later use
        all_labels.extend(labels.numpy())
        indices.extend([i * unlabeled_data.batch_size + j for j in range(len(images))])

    np.random.seed(seed=seed)
    selected_samples = np.random.choice(indices, size=num_samples, replace=False)
    selected_labels = [all_labels[idx] for idx in selected_samples]
    return selected_samples.tolist(), selected_labels

def pca_kmeans(unlabeled_train_data, num_samples,seed, pca_n_components=100):
    unlabeled_data = []
    indices = []
    all_labels = []
    for i, (images, labels) in enumerate(unlabeled_train_data):
        # Store the labels for later use
        unlabeled_data.extend(images.numpy())
        all_labels.extend(labels.numpy())
        indices.extend([i * unlabeled_train_data.batch_size + j for j in range(len(images))])


    unlabeled_data = np.array(unlabeled_data)
    pca = PCA(n_components=pca_n_components)
    pca_result = pca.fit_transform(unlabeled_data.reshape(len(unlabeled_data), -1))
    explained_varience = pca.explained_variance_ratio_
    print(f"Sum of explained variance in percent = {round(sum(explained_varience)*100,2)}")
    n_clusters = 2  # sick \ healthy
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(pca_result)
    centroids = kmeans.cluster_centers_

    # Calculate distances from each data point to the centroids
    distances = np.zeros((len(pca_result), n_clusters))
    for i, point in enumerate(pca_result):
        for j, centroid in enumerate(centroids):
            distances[i, j] = np.linalg.norm(point - centroid)
    ratios = distances[:, 0] / distances[:, 1]
    selected_samples = np.argsort(np.abs(ratios - 1))[:num_samples]
    selected_labels = [all_labels[idx] for idx in selected_samples]
    return selected_samples.tolist(), selected_labels
