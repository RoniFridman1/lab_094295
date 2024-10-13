from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans


def _pca_kmeans_sampling(unlabeled_train_data, num_samples, seed, pca_n_components=100):
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


def _bald_sampling(model, unlabeled_data, num_samples, mc_iterations=10):
    """
    Selects samples based on Bayesian Active Learning by Disagreement (BALD).

    Args:
        model (torch.nn.Module): Trained model used to select samples.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.
        mc_iterations (int): Number of Monte Carlo iterations for _uncertainty_sampling estimation.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model.train()  # Enable dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    uncertainties = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            batch_size = images.size(0)
            probs = torch.zeros(batch_size, 1, device=device)

            # Perform multiple forward passes
            for _ in range(mc_iterations):
                outputs = model(images)
                prob = torch.sigmoid(outputs)
                probs += prob

            probs /= mc_iterations  # Average probabilities

            # Compute mutual information
            entropy = -probs * torch.log(probs + 1e-20) - (1 - probs) * torch.log(1 - probs + 1e-20)
            expected_entropy = entropy.mean(dim=0)
            entropy_mean = entropy.mean(dim=0)
            mutual_info = entropy_mean - expected_entropy

            uncertainties.extend(mutual_info.cpu().numpy())
            all_indices.extend([batch_idx * unlabeled_data.batch_size + j for j in range(batch_size)])
            all_labels.extend(labels.numpy())

    uncertainties = np.array(uncertainties)
    all_indices = np.array(all_indices)

    # Select top samples with highest mutual information
    sorted_indices = np.argsort(uncertainties)[::-1]
    selected_indices = sorted_indices[:num_samples]
    selected_samples = all_indices[selected_indices]
    selected_labels = [all_labels[idx] for idx in selected_samples]

    return selected_samples.tolist(), selected_labels


def _margin_sampling(model, unlabeled_data, num_samples):
    """
    Selects samples based on Margin Sampling strategy.

    Args:
        model (torch.nn.Module): Trained model used to select samples.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    margins = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            outputs = model(images)  # Raw logits
            probabilities = torch.sigmoid(outputs)  # For binary classification

            # For binary classification, margin can be absolute difference from 0.5
            margins_batch = torch.abs(probabilities - 0.5).squeeze()
            margins.extend(margins_batch.cpu().numpy())
            all_indices.extend([batch_idx * unlabeled_data.batch_size + j for j in range(images.size(0))])
            all_labels.extend(labels.numpy())

    margins = np.array(margins)
    all_indices = np.array(all_indices)

    # Select samples with smallest margins (highest _uncertainty_sampling)
    sorted_indices = np.argsort(margins)
    selected_indices = sorted_indices[:num_samples]
    selected_samples = all_indices[selected_indices]
    selected_labels = [all_labels[idx] for idx in selected_samples]

    return selected_samples.tolist(), selected_labels


def _dpp_sampling(model, unlabeled_data, num_samples, similarity_threshold=0.5):
    """
    Selects samples based on Determinantal Point Processes (DPP) for diversity.

    Args:
        model (torch.nn.Module): Trained model used to extract features.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.
        similarity_threshold (float): Threshold for cosine similarity to ensure diversity.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    features = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            all_indices.extend([batch_idx * unlabeled_data.batch_size + j for j in range(images.size(0))])
            all_labels.extend(labels.numpy())

    features = np.vstack(features)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(features)

    # Simple greedy DPP-like selection
    selected_indices = []
    remaining_indices = set(range(len(features)))

    while len(selected_indices) < num_samples and remaining_indices:
        if not selected_indices:
            # Select the sample with the highest norm (most confident)
            norms = np.linalg.norm(features, axis=1)
            selected = np.argmax(norms)
        else:
            # Select the sample with the lowest similarity to already selected samples
            sim_to_selected = similarity_matrix[list(selected_indices)][:, list(remaining_indices)].mean(axis=0)
            selected = list(remaining_indices)[np.argmin(sim_to_selected)]

        selected_indices.append(selected)
        remaining_indices.remove(selected)

        # Remove samples that are too similar
        to_remove = [idx for idx in remaining_indices if similarity_matrix[selected][idx] > similarity_threshold]
        remaining_indices -= set(to_remove)

    selected_labels = [all_labels[idx] for idx in selected_indices]

    return selected_indices, selected_labels
