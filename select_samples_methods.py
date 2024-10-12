from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans


def select_samples(model, unlabeled_data, config, strategy='uncertainty', num_samples=10):
    """
    Selects the most informative samples based on the chosen strategy.

    Args:
        model (torch.nn.Module): Trained model used to select samples.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        strategy (str): Strategy for selecting samples ('_uncertainty_sampling', '_entropy_sampling', '_random_sampling').
        num_samples (int): Number of samples to select.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    if strategy == 'uncertainty':
        return _uncertainty_sampling(model, unlabeled_data, num_samples)
    elif strategy == 'entropy':
        return _entropy_sampling(model, unlabeled_data, num_samples)
    elif strategy == 'random':
        return _random_sampling(model, unlabeled_data, num_samples, seed=config.seed)
    elif strategy == "pca_kmeans":
        return _pca_kmeans_sampling(unlabeled_data, num_samples, seed=config.seed, pca_n_components=100)
    elif strategy == 'bald':
        return _bald_sampling(model, unlabeled_data, num_samples, mc_iterations=config.mc_iterations)
    elif strategy == 'margin':
        return _margin_sampling(model, unlabeled_data, num_samples)
    elif strategy == 'core_set':
        return _core_set_sampling(model, unlabeled_data, num_samples, config)
    elif strategy == 'dpp':
        return _dpp_sampling(model, unlabeled_data, num_samples)
    else:
        raise ValueError(f"Strategy {strategy} not recognized.")


def _entropy_sampling(model, unlabeled_data, num_samples):
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
            all_labels.extend(labels.numpy())  # Store the labels for later use
            entropy_scores = -torch.sum(probabilities * torch.log(probabilities + 1e-20),
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


def _uncertainty_sampling(model, unlabeled_data, num_samples):
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


def _random_sampling(model, unlabeled_data, num_samples, seed=42):
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


# todo:_pca_kmeans_sampling and core set are the same - I think that _core_set_sampling is the way it should be but need
#  to do tests.
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


def _core_set_sampling(model, unlabeled_data, num_samples, config):
    """
    Selects samples based on Core-Set selection strategy using k-means clustering.

    Args:
        model (torch.nn.Module): Trained model used to extract features.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.
        pca_n_components (int): Number of PCA components for dimensionality reduction.

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
            # Assume the feature extractor is up to the penultimate layer
            # Modify this line based on your model's architecture
            if hasattr(model, 'fc'):
                features_batch = model.fc.weight.data.cpu().numpy()
            else:
                # For models like VGG, modify accordingly
                features_batch = model.classifier[-1][1].weight.data.cpu()
            features.append(features_batch.cpu().numpy())
            all_indices.extend([batch_idx * unlabeled_data.batch_size + j for j in range(images.size(0))])
            all_labels.extend(labels.numpy())

    features = np.vstack(features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=config.PCA_N_COMPONENTS)
    reduced_features = pca.fit_transform(features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_samples, random_state=42)
    kmeans.fit(reduced_features)
    cluster_centers = kmeans.cluster_centers_

    # Select one sample per cluster
    selected_samples = []
    for center in cluster_centers:
        distances_to_center = np.linalg.norm(reduced_features - center, axis=1)
        closest_sample = np.argmin(distances_to_center)
        selected_samples.append(closest_sample)

    selected_labels = [all_labels[idx] for idx in selected_samples]

    return selected_samples, selected_labels


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
