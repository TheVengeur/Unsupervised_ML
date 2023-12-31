import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time

def Exercice2():

    # Load the data and labels
    data = np.load('Exercice2/data.npy')
    labels = np.load('Exercice2/labels.npy')

    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Perform PCA for 2D and 3D projections
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)

    data_2d = pca_2d.fit_transform(data_standardized)
    data_3d = pca_3d.fit_transform(data_standardized)

    # Create scatter plot for 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', edgecolors='k', s=40)
    plt.title('2D Projection of Meteorological Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Create scatter plot for 3D projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', s=40, edgecolors='k')
    ax.set_title('3D Projection of Meteorological Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # Add a legend
    legend_labels = ['No Tempest', 'Tempest']
    legend = ax.legend(legend_labels, loc='upper right')
    legend.legendHandles[0]._sizes = [30]


    plt.show()

if __name__ == '__main__':
    print("[EX2] Cluster and heuristic...")
    start = time()
    Exercice2()
    print(f"[EX2] Done - {(time() - start):0.4f}s")