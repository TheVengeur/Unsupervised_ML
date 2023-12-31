import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from time import time

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

def Exercice3():

    # Load the dataset
    data = np.load('Exercice3/data.npy')

    # Standardize the dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Function to determine the optimal number of clusters using the Elbow Method
    def elbow_method(data, max_clusters=10):
        distortions = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        return distortions

    # Function to determine the optimal number of clusters using the Silhouette Score
    def silhouette_analysis(data, max_clusters=10):
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhouette_scores.append(silhouette_avg)
        return silhouette_scores

    # Function to perform clustering using KMeans
    def perform_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        return labels

    # Function to perform clustering using Agglomerative Clustering
    def perform_agglomerative(data, n_clusters):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(data)
        return labels

    # Perform Elbow Method and Silhouette Analysis
    max_clusters = 10
    distortions = elbow_method(data_scaled, max_clusters)
    silhouette_scores = silhouette_analysis(data_scaled, max_clusters)

    # Plot Elbow Method
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    # Choose the optimal number of clusters based on the analysis
    optimal_clusters_kmeans = np.argmin(distortions) + 1  # Elbow Method
    optimal_clusters_agglomerative = np.argmax(silhouette_scores) + 2  # Silhouette Analysis

    # Perform clustering with optimal number of clusters
    labels_kmeans = perform_kmeans(data_scaled, optimal_clusters_kmeans)
    labels_agglomerative = perform_agglomerative(data_scaled, optimal_clusters_agglomerative)

    # Plot the clustered data
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels_kmeans, cmap='viridis', edgecolor='k', s=40)
    plt.title(f'KMeans Clustering (k={optimal_clusters_kmeans})')

    plt.subplot(1, 2, 2)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels_agglomerative, cmap='viridis', edgecolor='k', s=40)
    plt.title(f'Agglomerative Clustering (k={optimal_clusters_agglomerative})')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("[EX3] Cluster and heuristic...")
    start = time()
    Exercice3()
    print(f"[EX3] Done - {(time() - start):0.4f}s")