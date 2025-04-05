import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.svm import OneClassSVM
import numpy as np
from scipy.stats.mstats import winsorize

df = pd.read_csv("x_train_final.csv")
df_y = pd.read_csv("y_train_final_j5KGWWK.csv")

# Extract the first principal component from the PCA results and reshape to a 2D array
# df = pd.read_csv("x_train_pca.csv")
# pc1 = df["pca_one"].to_numpy().reshape(-1, 1)

# Using only PC1 for clustering (reshaped already as pc1 from previous blocks)
# Apply KMeans with 2 clusters on PC1 and enable verbose output.
# kmeans_2 = KMeans(n_clusters=2, random_state=42, verbose=1)
# clusters = kmeans_2.fit_predict(pc1)

# Display cluster distribution
# cluster_counts = pd.Series(clusters).value_counts()
# print("Cluster distribution:\n", cluster_counts)

# Heuristically, we assume the minority cluster corresponds to outliers.
# Identify the cluster with the fewest members.
# outlier_cluster = cluster_counts.idxmin()
# print("Assumed outlier cluster:", outlier_cluster)

# Create outlier labels: assign -1 for outliers and 1 for inliers.
# clusters_outlier_label = np.where(clusters == outlier_cluster, -1, 1)

# Save the outlier labels into the dataframe
# df['cluster_outlier'] = clusters_outlier_label

# First version of outliers
# new_df = df[df['cluster_outlier']==1]
# new_y_df = df_y[df_y["Unnamed: 0"].isin(new_df["Unnamed: 0"])]
#
# new_df.to_csv('x_train_no_outlier.csv')
# new_y_df.to_csv('y_train_no_outlier.csv')

# Second version, we take out all trains that have at least one outlier
# outlier_train = df[df['cluster_outlier']==-1]['train'].drop_duplicates().to_list()
# new_df = df[~df['train'].isin(outlier_train)]
# new_y_df = df_y[df_y["Unnamed: 0"].isin(new_df["Unnamed: 0"])]
# new_df.to_csv('x_train_no_outlier_new.csv')
# new_y_df.to_csv('y_train_no_outlier_new.csv')

# Third version, winsorizing
def winsorize_dataframe(df, x):
    """
    Winsorizes all numeric columns in the dataframe.

    Parameters:
      df: pandas DataFrame containing the data.
      x: float, fraction of data to winsorize on the left tail and 2x on the right tail.
         For example, x=0.05 will winsorize the lower 5% and upper 10% of the data.

    Returns:
      A new DataFrame with winsorized numeric columns.
    """
    df_wins = df.copy()
    # add log scale

    # Loop through numeric columns
    for col in ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]:
        # winsorize with limits: (x, 2x)
        df_wins[col] = winsorize(df_wins[col], limits=(x, x))
    return df_wins

# Example usage:
x = 0.006  # 0.5% on the left and 1% on the right
df_winsorized = winsorize_dataframe(df, x)

# Optionally, inspect the winsorized data:
print(df_winsorized.describe())
new_y_df = df_y[df_y["Unnamed: 0"].isin(df_winsorized["Unnamed: 0"])]
df_winsorized.to_csv('x_train_winsorize_new.csv')
new_y_df.to_csv('y_train_winsorize_new.csv')