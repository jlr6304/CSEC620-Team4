import numpy as np
import sklearn
import functions
import PCA
import matplotlib.pyplot as plt


def create_clusters(data, k, l):
    """
    create clusters and find centroids of normalized data
    :param data: data to be clustered
    :param k: number of clusters to separate the data into
    :param l: loss threshold between interations to stop reclustering
    :return: a list of well-fit centroids
    """
    # 1. Choose k centroids c1, ..., ck at random.
    centroid_indexes = np.random.choice(len(data), k, replace=False)
    centroids = data[centroid_indexes]
    iter = 0
    loss = np.infty
    while True:
        clusters = []
        cluster_loss = []
        # 2. Assign each data point xi to its nearest centroid.
        distances = sklearn.metrics.pairwise.euclidean_distances(data, centroids)
        classification = np.argmin(distances, axis=1)
        for i in range(k):
            clusters.append(data[classification == i])
            # 3. Recompute the centroids cj by taking the average of all the data points assigned to the jth cluster.
            centroids[i, :] = np.mean(clusters[i], axis=0)
            cluster_loss.append(np.sum(sklearn.metrics.pairwise.euclidean_distances(clusters[i], [centroids[i]])))
            # 4. Repeat (2) and (3) until the algorithm converges; that is, the difference between L X on successive
            # iterations is below a predetermined threshold.
        # print(iter, loss, sum(cluster_loss), "\n", centroids)
        if iter > 0:
            if abs(loss - sum(cluster_loss)) < l:
                break
        loss = sum(cluster_loss)
        iter += 1
    return centroids


def run(training, testing, k, t, l):
    """
    run: run the k-means algorithm
    :param training: the training set
    :param testing: the testing set
    :param k: the number of nearest neighbors
    :param t: the distance threshold by which to classify a point as an anomaly
    :param l: the loss threshold required to stop re-clustering
    :return: a list of binary labels (1 for anomaly, 0 for normal)
    """
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
        if np.min(distances[i]) < t:
            labels.append(0)
        else:
            labels.append(1)
    return labels


def tune_hyperparameters():
    """
    tune_hyperparameters: function to test out multiple values for t and k and determine the combination of these
    values which yields the highest f1-score
    """
    training, testing, labels = functions.get_data(n_testing_samples=4000, attacks_ratio = .2)
    condensed_training, condensed_testing = PCA.reduce_dimensions(training, testing, 12)

    bold = lambda x: '\033[1m' + x + '\033[0m'

    t_range = [.1,.11,.12,.13,.14,.15,.16,.17,.18,.19]
    k_range = [1, 2 ,3,4,5,6,7,8,9,10]
    print(bold("k-Means anomaly detection"))
    parameters = []
    results = []
    # generate array of f1-score results
    for t in t_range:
        for k in k_range:
            print(f"k: {k}\nt: {t}")
            parameters.append((k, t))
            Kmeans_labels = run(condensed_training, condensed_testing, k, t, l=.01)
            results.append(functions.score(Kmeans_labels, labels, ["F1score"])["F1score"])
    index = np.argmax(results)
    ideal_parameters = parameters[index]
    # display ideal parameters
    print(ideal_parameters, np.max(results))
    ideal_k = ideal_parameters[0]
    ideal_t = ideal_parameters[1]
    # get 1-D results for each hyperparameter
    t_results = []
    k_results = []
    for i in range(len(results)):
        if ideal_t in parameters[i]:
            k_results.append(results[i])
        if ideal_k in parameters[i]:
            t_results.append(results[i])
    plt.figure()
    # Plot each projection
    plt.scatter(t_range, t_results, c='b', edgecolors='none')
    # Graphical changes
    plt.xlabel("t")
    plt.ylabel("F1-score")
    plt.show()
    # Plot each projection
    plt.figure()
    plt.scatter(k_range, k_results, c='r', edgecolors='none')
    # Graphical changes
    plt.xlabel("k")
    plt.ylabel("F1-score")
    plt.show()


if __name__ == "__main__":
    tune_hyperparameters()

