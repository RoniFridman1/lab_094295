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
        config (Config): model's configuration
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
    elif strategy == 'core_set':
        return _core_set_sampling(model, unlabeled_data, num_samples, config)
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


def _core_set_sampling(model, unlabeled_data, num_samples, config):
    """
    Selects samples based on Core-Set selection strategy using k-means clustering.

    Args:
        model (torch.nn.Module): Trained model used to extract features.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.
        config (Config): the experiment's configuration.

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
            # Process one image at a time
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)  # Create a batch of size 1
                _ = model(image)
                features_batch = model.classifier[-1][1].weight.data.cpu().numpy()
                features.append(features_batch)
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
