import PCA
import Kmeans
import DBSCAN
from functions import get_data, score
import time

def main():
    # load and preprocess data
    training, testing, labels = get_data(n_testing_samples=4000, attacks_ratio = .2)
    condensed_training, condensed_testing = PCA.reduce_dimensions(training, testing, 10)
    
    # DBSCAN anomaly detection
    DBSCAN_start = time.time()
    DBSCAN_labels = DBSCAN.run(training, testing, min_neighbors, epsilon)
    DBSCAN_end = time.time()
    
    # Kmeans anomaly detection
    Kmeans_start = time.time()
    Kmeans_labels = Kmeans.run(training, testing, k, t)
    Kmeans_end = time.time()
    
    # Performances comparison
    print(DBSCAN_end-DBSCAN_start)
    print(Kmeans_end-Kmeans_start)
    score(DBSCAN_labels, labels)
    score(Kmeans_labels, labels)


if __name__ == "__main__":
    main()