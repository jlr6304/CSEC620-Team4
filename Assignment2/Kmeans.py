import numpy as np
import sklearn


def create_clusters(data, k, l):
    # 1. Choose k centroids c1, ..., ck at random.
    centroid_indexes = np.random.choice(len(data), k, replace=False)
    centroids = data[centroid_indexes]
    loss = l**2
    while True:
        clusters = []
        cluster_loss = []
        # 2. Assign each data point xi to its nearest centroid.
        distances = sklearn.metrics.pairwise.euclidean_distances(data, centroids)
        classification = np.argmin(distances, axis=1)
        for i in range(k):
            clusters.append(data[classification == i])
            # 3. Recompute the centroids cj by taking the average of all the data points assigned to the jth cluster.
            centroids[i, :] = np.mean(clusters[i])
            cluster_loss.append(np.sum(sklearn.metrics.pairwise.euclidean_distances(clusters[i], [centroids[i]])))
            # 4. Repeat (2) and (3) until the algorithm converges; that is, the difference between L X on successive
            # iterations is below a predetermined threshold.
        if loss - sum(cluster_loss) < l:
            break
        loss = sum(cluster_loss)
    return centroids


def run(training, testing, k, t, l):
    labels = []
    # 1. Use k-Means clustering to identify the centroids of the clusters in the normal traffic
    # training set.
    centroids = create_clusters(training, k, l)
    # 2. Select a distance threshold value, t.
    distances = sklearn.metrics.pairwise.euclidean_distances(testing, centroids)
    # 3. For each test sample, find the cluster centroid to which the sample is closest using a
    # distance function (e.g. Euclidean distance).
    # If the distance is less than your threshold value t,
    # then classify the sample as normal. If it is greater than your threshold value t,
    # then classify the sample as anomalous.
    for i in range(len(distances)):
        if np.argmin(distances[i]) < t:
            labels.append(0)
        else:
            labels.append(1)
    return labels


