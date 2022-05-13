import numpy as np
import idx2numpy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn import cluster
from kmeans import run_kmeans, rand_index
from sklearn.cluster import KMeans

# load images from MNIST
train_file = "mnist/train-images-idx3-ubyte"
train_label_file = "mnist/train-labels-idx1-ubyte"
raw_images = idx2numpy.convert_from_file(train_file)
raw_labels = idx2numpy.convert_from_file(train_label_file)

# plt.imshow(images[0], cmap='Greys')
# plt.show()

k = 10

# convert images from 28x28 to flat arrays
images = np.array([x.flatten() for x in raw_images])

# take random sample
num_samples = 4000
idxs = np.random.choice(int(images.shape[0]), num_samples)
image_subset = images[idxs, :]
label_subset = raw_labels[idxs]

# adjacency matrix using gaussian kernel
neighbors_params = [640]
# neighbors_params = [80, 160, 320, 640, 1280]
for neighbors in neighbors_params:
    # adjacency matrix using kmeans
    distances = squareform(pdist(image_subset, 'sqeuclidean'))
    W = []
    for row in distances:
        # get the nearest neighbors for each image
        nearest_images = np.argpartition(row, neighbors)
        # set the nearest neighbors to 1 and everything else to 0
        row[nearest_images[:neighbors]] = 1
        row[nearest_images[neighbors:]] = 0
        W.append(row)
    W = np.array(W)
    # degree matrix
    D = np.diag(np.sum(W, axis=1))
    # laplacian
    L = D - W
    # last k get eigenvalues
    values, vectors = np.linalg.eigh(L) 
    top_k_idxs = np.argsort(values)[1:k]
    X = vectors[:,top_k_idxs]
    # run kmeans on the projected data
    sk_kmeans = KMeans(n_clusters=k)
    sk_kk_labels = sk_kmeans.fit_predict(X)
    rand_score = rand_index(sk_kk_labels, label_subset)
    print("Neighbors: ", neighbors, ". Rand score: ", rand_score)
    print(np.bincount(sk_kk_labels))

# # adjacency matrix using gaussian kernel
# sigmas = [0.01, 0.1, 1, 100, 1000, 10000, 100000]
# for sigma in sigmas:
#     distances = squareform(pdist(image_subset, 'sqeuclidean'))
#     W = np.exp(-distances / sigma)
#     # degree matrix
#     D = np.diag(np.sum(W, axis=1))
#     # laplacian
#     L = D - W
#     # last k get eigenvalues
#     values, vectors = np.linalg.eigh(L) 
#     top_k_idxs = np.argsort(values)[1:k]
#     X = vectors[:,top_k_idxs]
#     # run kmeans on the projected data
#     sk_kmeans = KMeans(n_clusters=k)
#     sk_kk_labels = sk_kmeans.fit_predict(X)
#     rand_score = rand_index(sk_kk_labels, label_subset)
#     print("Sigma: ", sigma, ". Rand score: ", rand_score)
#     print(np.bincount(sk_kk_labels))

unique_clusters = np.unique(sk_kk_labels)
fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
for c in unique_clusters:
    idxs = np.where(sk_kk_labels == c)[0]
    cluster = np.take(image_subset, idxs, axis=0)
    centroid = np.around(np.mean(cluster, axis=0))
    digit = np.reshape(centroid, (28,28))
    ax[c].imshow(digit, cmap='Greys')
    ax[c].set_yticklabels([])
    ax[c].set_xticklabels([])
    ax[c].set_xticks([])
    ax[c].set_yticks([])

plt.show()

unique_clusters = np.unique(label_subset)
fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
for c in unique_clusters:
    idxs = np.where(label_subset == c)[0]
    cluster = np.take(image_subset, idxs, axis=0)
    centroid = np.around(np.mean(cluster, axis=0))
    digit = np.reshape(centroid, (28,28))
    ax[c].imshow(digit, cmap='Greys')
    ax[c].set_yticklabels([])
    ax[c].set_xticklabels([])
    ax[c].set_xticks([])
    ax[c].set_yticks([])

plt.show()