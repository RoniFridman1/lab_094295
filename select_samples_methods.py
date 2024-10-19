import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from visualization import _plot_clusters_3d


def select_samples(iter_model, unlabeled_data, config, strategy='uncertainty', num_samples=10):
    """
    Selects the most informative samples based on the chosen strategy.

    Args:
        iter_model (torch.nn.Module): Trained model used to select samples.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        config (Config.py): model's configuration
        strategy (str): Strategy for selecting samples (uncertainty, entropy, random etc).
        num_samples (int): Number of samples to select.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model = iter_model.model
    if strategy == 'uncertainty':
        return _uncertainty_sampling(model, unlabeled_data, num_samples)
    elif strategy == 'entropy':
        return _entropy_sampling(model, unlabeled_data, num_samples)
    elif strategy == 'random':
        return _random_sampling(model, unlabeled_data, num_samples, seed=config.seed)
    elif strategy == 'pca_then_kmeans':
        return _pca_then_kmeans_sampling(iter_model, unlabeled_data, num_samples, config=config)
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


def _pca_then_kmeans_sampling(iter_model, unlabeled_data, num_samples, config):
    """
    Selects samples based on the following strategy:
    1. Extract for each sample the weights of the last layer of the model before the clustering itself (a vector of
       weights for each sample).
    2. Perform PCA to the samples' weights' vectors and lower the dimension to a relatively low number (eg. 3 or 10).
    3. perform K-Means and split the unlabeled samples to K=num_samples clusters.
    4. Select one image from every cluster to label.
    The underlying assumption that different clusters contain images from with different characteristics, and it will be
    valuable to select images with diverse characteristics fro the training.

    Args:
        iter_model (torch.nn.Module): Trained model used to extract features.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        num_samples (int): Number of samples to select.
        config (Config.py): the experiment's configuration.

    Returns:
        selected_samples (list): Indices of selected samples.
        selected_labels (list): Corresponding labels of the selected samples.
    """
    model = iter_model.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    features = []
    all_indices = []
    all_labels = []
    with torch.no_grad():
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        for batch_idx, (images, labels) in enumerate(unlabeled_data):
            images = images.to(device)
            # Process one image at a time
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)  # Create a batch of size 1
                # Last layer is sequential of Dropout->Linear. So we take the weights of its second internal layer.
                features_batch = feature_extractor(image).reshape(-1).cpu().numpy()
                features.append(features_batch)
            all_indices.extend([batch_idx * unlabeled_data.batch_size + j for j in range(images.size(0))])
            all_labels.extend(labels.numpy())

    features = np.vstack(features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=config.PCA_N_COMPONENTS)
    reduced_features = pca.fit_transform(features)
    explained_deviance = pca.explained_variance_ratio_
    print(f"Total explain variance ratio: {round(sum(explained_deviance)*100,2)}")

    # Perform k-means clustering, and choose best clusters using silhouette scores
    kmeans = KMeans(n_clusters=config.K_CLUSTERS, random_state=config.seed)
    kmeans.fit(reduced_features)
    cluster_labels = kmeans.labels_

    # Compute silhouette scores for each cluster and choose top clusters to sample from.
    cluster_scores = silhouette_samples(reduced_features, cluster_labels)
    avg_cluster_scores = [np.mean(cluster_scores[cluster_labels == i]) for i in range(config.K_CLUSTERS)]
    top_clusters = np.argsort(avg_cluster_scores)[-num_samples:]

    # Filter cluster centers based on top clusters
    cluster_centers = [kmeans.cluster_centers_[i] for i in top_clusters]

    if config.PCA_N_COMPONENTS == 3:  # Plot if PCA is 3 dimensions
        _plot_clusters_3d(reduced_features, cluster_labels, top_clusters)
    # Select one sample per cluster
    selected_samples = []
    for center in cluster_centers:
        distances_to_center = np.linalg.norm(reduced_features - center, axis=1)
        closest_sample = np.argmin(distances_to_center)
        selected_samples.append(closest_sample)

    selected_labels = [all_labels[idx] for idx in selected_samples]

    return selected_samples, selected_labels
