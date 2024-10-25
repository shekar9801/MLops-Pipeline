import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def perform_pca_2d(data):
    try:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        return principal_components
    except Exception as e:
        raise RuntimeError(f'Error during PCA: {str(e)}')

def perform_knn_classification(data, labels, n_neighbors=3):
    """
    Perform K-Nearest Neighbors classification on the data.
    
    Parameters:
        data (array-like): The data to classify.
        labels (array-like): The labels corresponding to the data.
        n_neighbors (int): The number of neighbors to consider.
        
    Returns:
        predicted_labels (ndarray): Predicted labels for each point in the data.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(data, labels)
    predicted_labels = knn.predict(data)
    return predicted_labels

def plot_2d_pca_with_clusters(principal_components, cluster_labels):
    plt.figure(figsize=(8, 6))
    
    # Get unique cluster labels
    unique_labels = np.unique(cluster_labels)
    
    # Define a color map
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        # Get the indices of the points in the current cluster
        cluster_points = principal_components[cluster_labels == label]
        
        # Plot each cluster with a solid color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors(i), label=f'Cluster {label}', alpha=0.7)
    
    plt.title("2D PCA with KNN Classification")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.colorbar(label='Cluster Label')
    plt.legend()
    plt.show()