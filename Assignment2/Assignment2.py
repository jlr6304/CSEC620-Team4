import PCA
import Kmeans
import DBSCAN
from functions import get_data, score
import time

def main():
    training, testing, labels = get_data()
    condensed_training, condensed_testing = PCA.reduce_dimensions(testing, training, 5)
    DBSCAN_start = time.time()
    DBSCAN_labels = DBSCAN.run(training, testing, min_neighbors, epsilon)
    DBSCAN_end = time.time()
    Kmeans_start = time.time()
    Kmeans_labels = Kmeans.run(training, testing, k, t)
    Kmeans_end = time.time()
    print(DBSCAN_end-DBSCAN_start)
    print(Kmeans_end-Kmeans_start)
    score(DBSCAN_labels, labels)
    score(Kmeans_labels, labels)

    pass

if __name__ == "__main__":
    main()