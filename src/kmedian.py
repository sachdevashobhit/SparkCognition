import numpy as np
from scipy.spatial.distance import cdist


def evaluate_centroids(data, clusters, k):
    # k = np.max(clusters) + 1
    centroids = np.empty(shape=(k,) + data.shape[1:])
    for i in range(k):
        np.mean(data[clusters == i], axis=0, out=centroids[i])
    centroids = np.nan_to_num(centroids)
    return centroids    


def kmedian(data, k, steps=10):

    # case: len(data) <= k
    if len(data) == k:
        return data
    elif len(data) == 0:
        return np.zeros(shape=(1, 1440))
    elif len(data) < k:
        return np.concatenate((data, np.tile(data[0], [k-len(data), 1])), axis=0)

    # randomly pick k data points as centroids
    centroids = data[np.random.choice(np.arange(len(data)), k, False)]
    print 'centroids', centroids
    
    for _ in range(max(steps, 1)):
        # Squared distances between each point and each centroid.
        # sqdists = scipy.spatial.distance.cdist(centroids, data, 'sqeuclidean')
        # print sqdists, 'sqdists'

        # Manhattan distance
        mandist = cdist(centroids, data, 'cityblock')

        # Index of the closest centroid to each data point.
        clusters = np.argmin(mandist, axis=0)
        # print clusters, 'clusters'

        new_centroids = evaluate_centroids(data, clusters, k)
        # print new_centroids, 'new_centroids'
        
        if np.array_equal(new_centroids, centroids): break
        
        centroids = new_centroids
        # print 'centroids', centroids
        
    return centroids