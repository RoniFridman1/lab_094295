from data_loader import load_data
from Config import Config
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans


# load the data according to the configuration
cnfg = Config()
_, train_loader_unlabeled, _, _ = load_data(
    cnfg.DATA_DIR, batch_size=cnfg.BATCH_SIZE, labeled_unlabeled_split=cnfg.TRAIN_LABELED_UNLABELED_RATIO,
    total_train_samples=cnfg.TOTAL_TRAINING_SAMPLES, total_test_samples=cnfg.TOTAL_TEST_SAMPLES,
    seed=cnfg.seed)

unlabeled_data = []
for batch in train_loader_unlabeled:
    images, _ = batch
    unlabeled_data.extend(images.numpy())

# Convert the data to a numpy array
unlabeled_data = np.array(unlabeled_data)

# Run PCA dimension reduction
pca = PCA(n_components=100)
pca_result = pca.fit_transform(unlabeled_data.reshape(len(unlabeled_data), -1))
ev = pca.explained_variance_ratio_
print(sum(ev))
# Plot the PCA results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], cmap='viridis')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title("K-Means Clustering on PCA-Reduced Data (3D)")
plt.show()


# Create a K-Means clustering model
n_clusters = 2  # sick \ healthy
kmeans = KMeans(n_clusters=n_clusters, random_state=cnfg.seed)

# Fit the model to the PCA-reduced data
cluster_labels = kmeans.fit_predict(pca_result)

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_labels, cmap='viridis')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title("K-Means Clustering on PCA-Reduced Data (3D)")
plt.show()

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Calculate distances from each data point to the centroids
distances = np.zeros((len(pca_result), n_clusters))
for i, point in enumerate(pca_result):
    for j, centroid in enumerate(centroids):
        distances[i, j] = np.linalg.norm(point - centroid)

# Print the distance matrix
print("Distance Matrix:")
print(distances)

# Calculate the ratio of left column to right column
ratios = distances[:, 0] / distances[:, 1]

# Find the indices of the 25 rows with the closest ratio to 1
closest_ratio_indices = np.argsort(np.abs(ratios - 1))[:cnfg.SAMPLES_PER_ITERATION]

print("Indices of the 25 rows with the closest ratio to 1:", closest_ratio_indices)




