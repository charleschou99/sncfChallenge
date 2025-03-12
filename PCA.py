import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import numpy as np

df = pd.read_csv("x_train_pca.csv")


# Extract the first principal component from the PCA results and reshape to a 2D array
pc1 = df["pca_one"].to_numpy().reshape(-1, 1)

# Using only PC1 for clustering (reshaped already as pc1 from previous blocks)
# Apply KMeans with 2 clusters on PC1 and enable verbose output.
kmeans_2 = KMeans(n_clusters=2, random_state=42, verbose=1)
clusters = kmeans_2.fit_predict(pc1)

# Display cluster distribution
cluster_counts = pd.Series(clusters).value_counts()
print("Cluster distribution:\n", cluster_counts)

# Heuristically, we assume the minority cluster corresponds to outliers.
# Identify the cluster with the fewest members.
outlier_cluster = cluster_counts.idxmin()
print("Assumed outlier cluster:", outlier_cluster)

# Create outlier labels: assign -1 for outliers and 1 for inliers.
clusters_outlier_label = np.where(clusters == outlier_cluster, -1, 1)

# Save the outlier labels into the dataframe
df['cluster_outlier'] = clusters_outlier_label

# Plot PC1 with outlier detection via KMeans classification
plt.figure(figsize=(10, 6))
plt.scatter(range(len(pc1)), pc1, c=clusters_outlier_label, cmap='coolwarm', alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("First Principal Component (PC1)")
plt.title("Outlier Detection using KMeans (2 Clusters) on PC1")
plt.colorbar(label="Outlier Label (-1: Outlier, 1: Inlier)")
plt.show()
