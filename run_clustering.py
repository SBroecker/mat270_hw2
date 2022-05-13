import numpy as np
import idx2numpy
from kmeans import run_kmeans, rand_index
import matplotlib.pyplot as plt

# load images from MNIST
train_file = "mnist/train-images-idx3-ubyte"
train_label_file = "mnist/train-labels-idx1-ubyte"
mnist_images = idx2numpy.convert_from_file(train_file)
mnist_labels = idx2numpy.convert_from_file(train_label_file)

# convert images from 28x28 to flat arrays
images = np.array([x.flatten() for x in mnist_images])

# assign number of clusters
k = 10

cluster_assignments, cluster_centroids, updates = run_kmeans(images, 0, k)

rand_score = rand_index(cluster_assignments, mnist_labels)
print("Rand score: ", rand_score)

fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
for index, c in enumerate(cluster_centroids):
    digit = np.reshape(c, (28,28))
    ax[index].imshow(digit, cmap='Greys')
    ax[index].set_yticklabels([])
    ax[index].set_xticklabels([])
    ax[index].set_xticks([])
    ax[index].set_yticks([])

plt.show()

