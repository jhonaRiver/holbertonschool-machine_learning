#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

plt.subplot(1, 1, 1, projection='3d')
plt.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
            c=labels, cmap=plt.get_cmap('plasma'),
            label=labels)
plt.xlabel('U1')
plt.ylabel('U2')
plt.zlabel('U3')
plt.title('PCA of Iris Dataset')
plt.show()
