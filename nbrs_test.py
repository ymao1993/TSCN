import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

dataset = np.array(
    [[1., 1, 1, 2, 3],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
testset = np.array(
    [[1., 1, 1, 1, 1],
     [90, 90, 10, 10, 1]
     ])

print('NearestNeighbors')
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(dataset)
distances, indices = nbrs.kneighbors(dataset)
print(distances)
print(indices)

print('cKDTree')
kdtree = cKDTree(dataset, 3)
distances, indices = kdtree.query(dataset, k=2)
print(distances)
print(indices)
