from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from diffusion_map import diffusion_map, diffusion_matrix
import pandas as pd
import plotly.express as px

# unpack .mat file -> output is a dictionary
mat_contents = loadmat("genomedata.mat")

# get data from the X column
raw_data = mat_contents["X"]

# get rid of all the list nesting
unnested_data = np.array([x[0][0] for x in raw_data])

# split on tab characters
split_data = np.array([x.split("\t") for x in unnested_data])

# get rid of whitespace
# get rid of space in last column
data = np.array([x.strip() for row in split_data for x in row]).reshape((5000, 1044))
data = np.delete(data,-1,1)

# convert the strings to numbers
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_data = enc.fit_transform(data)


scores = []
sigmas = [10]
for sigma in sigmas:
    # create a diffusion map
    S, D_left, values, vectors = diffusion_matrix(encoded_data, sigma)
    for n_comp in range(1, 20, 2):
        # get the number of components of the map
        d_map = diffusion_map(values, vectors, D_left, n_comp)
        for n_clust in range(2, 20, 2):
            # cluster the data based on the map
            clusterer = KMeans(n_clusters=n_clust)
            sk_labels = clusterer.fit_predict(d_map)
            # check how many items are in each cluster
            bins = np.bincount(sk_labels)
            # metric to check clustering effectiveness
            score = metrics.silhouette_score(d_map, sk_labels, metric='euclidean')
            scores.append([sigma, n_comp, n_clust, score, bins])
            print([sigma, n_comp, n_clust, score, bins])

# save the scores for all configurations
df = pd.DataFrame(scores, columns=["sigma", "components", "clusters", "score", "bins"])
df.to_csv("jacard_tests.csv")

# # plot
# df = pd.read_csv("pca_tests.csv", index_col=0)
# for name, group in df.groupby("sigma"):
#     fig = px.line(group, x="clusters", y="score", color="components", title=name)
#     fig.update_layout(font=dict(size=18))
#     fig.show()

# build 3d plot for report
S, D_left, values, vectors = diffusion_matrix(encoded_data, 10)
d_map = diffusion_map(values, vectors, D_left, 3)
clusterer = KMeans(n_clusters=4)
sk_labels = clusterer.fit_predict(d_map)

labeled_clusters = np.hstack([d_map*1000, sk_labels[:, np.newaxis]])

fig = px.scatter_3d(labeled_clusters, x=[0]*len(labeled_clusters), y=1, z=2, color=3)
fig.show()

# test dimension reduction with PCA
scores = []
for n_comp in range(1, 20, 2):
    for n_clust in range(2, 20, 2):
        pca = PCA(n_components=n_comp)
        pca_data = pca.fit_transform(encoded_data)
        clusterer = KMeans(n_clusters=n_clust)
        sk_labels = clusterer.fit_predict(pca_data)
        score = metrics.silhouette_score(pca_data, sk_labels, metric='euclidean')
        scores.append([n_comp, n_clust, score])
        # print(metrics.calinski_harabasz_score(pca_data, sk_labels))

df = pd.DataFrame(scores, columns=["components", "clusters", "score"])

fig = px.line(df, x="clusters", y="score", color="components")
fig.update_layout(font=dict(size=18))
fig.show()

# build 3d plot for report
pca = PCA(n_components=3)
pca_data = pca.fit_transform(encoded_data)
clusterer = KMeans(n_clusters=4)
sk_labels = clusterer.fit_predict(pca_data)
labeled_clusters = np.hstack([pca_data, sk_labels[:, np.newaxis]])
fig = px.scatter_3d(labeled_clusters, x=0, y=1, z=2, color=3)
fig.update_layout(font=dict(size=18))
fig.show()
