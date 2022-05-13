import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def diffusion_matrix(X, sigma):
    # get weight matrix
    distances = 1-squareform(pdist(X, 'jaccard'))
    # W = np.exp(-distances / sigma)
    # W = []
    # for row in distances:
    #     nearest_images = np.argpartition(row, sigma)
    #     row[nearest_images[:sigma]] = 1
    #     row[nearest_images[sigma:]] = 0
    #     W.append(row)
    # W = np.array(W)

    W = distances
    # calculate inverse degree matrix
    d = np.sum(W, axis=1)
    Di = np.diag(1/d)

    # get P matrix
    P = np.matmul(Di, W)
    
    # get left and right matrices
    D_right = np.diag((d)**0.5)
    D_left = np.diag((d)**-0.5)

    # get symmetric matrix from Ds and P
    S = np.matmul(D_right, np.matmul(P,D_left))

    # get eigenstuff for S matrix
    values, vectors = eigh(S)

    return S, D_left, values, vectors

def diffusion_map(values, vectors, D_left, k):
    # get top k eigenvectors based on values
    idx = values.argsort()[::-1]
    vectors = vectors[:,idx]
    
    # create diffusion map from D and eigenvectors
    diffusion_map = np.matmul(D_left, vectors)
    
    return diffusion_map[:,:k]