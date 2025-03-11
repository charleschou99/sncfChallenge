import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from format_x import preprocess_df

def perform_PCA_x():
    # Load the training data
    df = pd.read_csv("x_train_final.csv")
    df_processed = preprocess_df(df)

    # Standardize the data for PCA (important when features have different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed)

    # Perform PCA (here we choose 10 components for illustration; adjust as needed)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # --- Plotting Explained Variance Ratio ---
    plt.figure(figsize=(10, 6))
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    plt.bar(components, pca.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xticks(components)
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png")
    plt.show()

    # --- Scatter Plot of the First Two Principal Components ---
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Projection on the First Two Principal Components')
    plt.tight_layout()
    plt.savefig("pca_scatter.png")
    plt.show()

    # Save the PCA-transformed data for potential future use
    pca_columns = [f'PC{i}' for i in components]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca.to_csv("x_train_pca.csv", index=False)
    print("PCA results saved to x_train_pca.csv")