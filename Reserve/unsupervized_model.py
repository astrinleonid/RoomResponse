import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns


def analyze_clusters(dataset_path='mfcc_dataset.csv'):
    """
    Build an unsupervised learning model on the MFCC dataset and analyze
    how the clusters correlate with the original labels.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")

    # Extract features and labels
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]
    X = df[feature_cols].values
    true_labels = df['label'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Also try t-SNE for better visualization of clusters
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Determine optimal number of clusters using silhouette scores
    silhouette_scores = []
    k_range = range(2, min(10, X.shape[0] // 5 + 1))  # Limit to reasonable range

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score: {score:.4f}")

    # Find optimal number of clusters
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate adjusted Rand index to measure clustering performance against true labels
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    # Create a confusion matrix
    conf_matrix = confusion_matrix(true_labels, cluster_labels)

    # Add results to DataFrame for easier analysis
    df['cluster'] = cluster_labels

    # Plot results
    plot_clustering_results(X_pca, X_tsne, true_labels, cluster_labels, silhouette_scores, k_range, conf_matrix)

    # Analyze how each cluster matches with original labels
    cluster_analysis = pd.crosstab(df['cluster'], df['label'], normalize='index') * 100
    print("\nCluster to label distribution (%):")
    print(cluster_analysis)

    # Try DBSCAN as an alternative clustering approach
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Count number of clusters found (-1 is noise)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"\nDBSCAN found {n_clusters_dbscan} clusters and {np.sum(dbscan_labels == -1)} noise points")

    # Calculate ARI for DBSCAN
    if n_clusters_dbscan > 1:  # Only calculate if more than one cluster found
        ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)
        print(f"DBSCAN Adjusted Rand Index: {ari_dbscan:.4f}")

    return df


def plot_clustering_results(X_pca, X_tsne, true_labels, cluster_labels, silhouette_scores, k_range, conf_matrix):
    """
    Plot the clustering results in multiple views.
    """
    fig = plt.figure(figsize=(20, 15))

    # 1. Plot silhouette scores
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(k_range, silhouette_scores, 'o-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs. Number of Clusters')
    ax1.grid(True)

    # 2. Plot PCA with true labels
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.7)
    ax2.set_title('PCA projection with true labels')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    plt.colorbar(scatter, ax=ax2, label='True Label')

    # 3. Plot PCA with cluster labels
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
    ax3.set_title('PCA projection with cluster labels')
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    plt.colorbar(scatter, ax=ax3, label='Cluster')

    # 4. Plot t-SNE with true labels
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.7)
    ax4.set_title('t-SNE projection with true labels')
    ax4.set_xlabel('t-SNE Component 1')
    ax4.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=ax4, label='True Label')

    # 5. Plot t-SNE with cluster labels
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
    ax5.set_title('t-SNE projection with cluster labels')
    ax5.set_xlabel('t-SNE Component 1')
    ax5.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=ax5, label='Cluster')

    # 6. Plot confusion matrix
    ax6 = plt.subplot(2, 3, 6)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax6)
    ax6.set_title('Confusion Matrix: True Labels vs Clusters')
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=300)
    plt.show()


# Main execution
if __name__ == "__main__":
    results_df = analyze_clusters('mfcc_dataset.csv')

    # Additional analysis
    print("\nSummary statistics by cluster:")
    print(results_df.groupby('cluster')[['mfcc_0', 'mfcc_1', 'mfcc_2']].mean())

    # Check if binary clusters match binary labels
    if len(set(results_df['cluster'])) == 2:
        # Map largest cluster of each label
        label_0_dominant_cluster = results_df[results_df['label'] == 0]['cluster'].mode()[0]
        label_1_dominant_cluster = results_df[results_df['label'] == 1]['cluster'].mode()[0]

        # Calculate accuracy if we use this mapping
        mapped_pred = results_df['cluster'].map({
            label_0_dominant_cluster: 0,
            label_1_dominant_cluster: 1
        })

        accuracy = (mapped_pred == results_df['label']).mean() * 100
        print(f"\nIf we map clusters to labels based on majority class:")
        print(f"Cluster {label_0_dominant_cluster} → Label 0")
        print(f"Cluster {label_1_dominant_cluster} → Label 1")
        print(f"Accuracy: {accuracy:.2f}%")