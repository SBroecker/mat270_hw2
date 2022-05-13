import numpy as np
from scipy.special import comb
from scipy.stats import mode
import math

def assign_centroids(images, centroids):
    # create a list to keep track of cluster assignments
    cluster_assignments = []
    # go through each image
    for image in images:
        # make a list to keep track of the distance from the image to each centroid
        distances = []
        for index, centroid in enumerate(centroids):
            # calculate the norm of the difference between the two
            distance = np.linalg.norm(image - centroid)
            # add that distance to our list
            distances.append([distance, index])       
        # get the centroid index for the smallest distace
        closest_centroid = min(distances)[1]
        # add the index of that centroid to the cluster assignment list
        cluster_assignments.append(closest_centroid)
    return np.array(cluster_assignments)

def calculate_centroids(images, cluster_assignments, k):
    # list to keep track of newly calculated centroids
    new_centroids = []
    # for each cluster label
    for x in range(k):
        # find indexes for each point in the cluster
        idxs = np.where(cluster_assignments == x)[0]
        # get points that correspond to those indexes
        cluster = np.take(images, idxs, axis=0)
        if len(cluster) == 0:
            # randomly assign a center if no points are assigned to the cluster
            centroid = np.random.uniform(low=math.floor(np.min(images)), high=math.ceil(np.max(images)), size=(1, images.shape[1]))
        else:
            # get the mean for each dimension
            centroid = np.around(np.mean(cluster, axis=0))
        # add the new centroid calculation to the list
        new_centroids.append(centroid)
    return np.array(new_centroids)

def run_kmeans(images, stopping_threshold, k):
    # randomly create k clusters with values from 0 to 256
    centroids = np.random.uniform(low=math.floor(np.min(images)), high=math.ceil(np.max(images)), size=(k, images.shape[1]))
    # randomly create an array with cluster assignments
    old_assignments = np.random.randint(low=0, high=k, size=len(images))
    # flag to stop algorithm when centoids stop moving
    converged = False
    # keeping track of iterations and changes
    iters = 1
    num_updates = []
    # iterate until convergence
    while not converged and iters < 100:
        print("Assigning images to clusters. Iteration ", iters)
        # assign clusters to the centroids
        cluster_assignments = assign_centroids(images, centroids)
        print("Calculating new centroids")
        # calculate new centroids based on cluster assignments
        centroids = calculate_centroids(images, cluster_assignments, k)
        # check if clusters have moved
        diff = old_assignments - cluster_assignments
        # the difference between assignments will be 0 for samples that didn't move
        moved = np.where(diff != 0)[0]
        updates = len(moved)
        # if the cluster assignments are different than the previous iteration
        if updates > stopping_threshold:
            print(updates, " images changed clusters since the last iteration")
            # set up for next loop
            old_assignments = cluster_assignments
            iters += 1
            num_updates.append(updates)
        else:
            print("Converged!")
            # flag that algorithm has converged
            converged = True
    print("Converged in {} iterations".format(str(iters)))
    return cluster_assignments, centroids, num_updates

def rand_index(predicted_assignments, true_assignments):
    # https://scikit-learn.org/stable/modules/clustering.html#rand-index
    # how many groups of 2 you can make with the elements of each cluster
    tp_plus_fp = comb(np.bincount(true_assignments), 2).sum()
    tp_plus_fn = comb(np.bincount(predicted_assignments), 2).sum()
    # combine true and predicted into single array
    A = np.c_[(true_assignments, predicted_assignments)]
    # get unique cluster indexes
    clusters = set(true_assignments)
    # A[:, 0] == i -> check if true assignment matches current cluster
    # A[A[:, 0] == i, 1] -> get elements from predicted assignments that are at those indexes
    # np.bincount(A[A[:, 0] == i, 1]) -> the number of elements assigned to each predicted cluster
    # comb(np.bincount(A[A[:, 0] == i, 1]), 2) -> the number of combinations of 2 you can make within each bin
    # step above is where true positives are calculated
    # comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() -> sum all of those combinations
    # comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in clusters -> do the same for all clusters
    # sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in clusters) -> sum across all clusters
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in clusters)
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def accuracy(predicted_assignments, true_assignments):
    accuracy = 0
    num_clusters = len(np.unique(predicted_assignments))
    for x in range(num_clusters):
        # get indexes for current cluster
        idxs = np.where(predicted_assignments == x)[0]
        # get labels that correspond to those indexes
        cluster_labels = np.take(true_assignments, idxs, axis=0)
        label_homogeneity = (cluster_labels == mode(cluster_labels)).sum() / len(cluster_labels)
        accuracy += label_homogeneity / num_clusters
    return accuracy