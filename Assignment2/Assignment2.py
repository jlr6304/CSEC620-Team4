import PCA
import Kmeans
import DBSCAN
from functions import get_data, score
import time

def main():
    # ----- load and preprocess data
    training, testing, labels = get_data(n_testing_samples=4000, attacks_ratio = .2)
    condensed_training, condensed_testing = PCA.reduce_dimensions(training, testing, 12)

    bold = lambda x: '\033[1m' + x + '\033[0m'

    # ----- DBSCAN anomaly detection
    print(bold("DBSCAN anomaly detection"))
    DBSCAN_start = time.time()
    DBSCAN_labels = DBSCAN.run(condensed_training, condensed_testing, min_neighbors=10, epsilon=.03, verbose=False)
    DBSCAN_end = time.time()
    
    # ----- Kmeans anomaly detection
    print(bold("k-Means anomaly detection"))
    Kmeans_start = time.time()
    Kmeans_labels = Kmeans.run(condensed_training, condensed_testing, k=8, t=.18, l=.01)
    Kmeans_end = time.time()
    
    # ----- Performances comparison
    # DBSCAN
    print(bold("DBSCAN performances"))
    print("Execution time:", DBSCAN_end - DBSCAN_start)
    score(DBSCAN_labels, labels, ["confusion_mat", "accuracy", "TPR", "recall", "FPR", "F1score"])
    
    # k-Means
    print(bold("k-Means performances"))
    print("Execution time:", Kmeans_end - Kmeans_start)
    score(Kmeans_labels, labels, ["confusion_mat", "accuracy", "TPR", "recall", "FPR", "F1score"])


if __name__ == "__main__":
    main()