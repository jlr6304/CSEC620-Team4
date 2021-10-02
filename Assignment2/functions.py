import numpy as np

import sklearn
import PCA
from matplotlib import pyplot as plt

def get_data(n_testing_samples=4000, attacks_ratio = .2):
    """
    ## Import data to cluster from a data repository

        `parameters`
            n_testing_samples: number of samples in the testing set
            attacks_ratio: ratio between the attack and normal samples in the testing set
        `return`
            training: training set
            testing: testing set 
            labels: labels of the samples in the testing set

    `Caution the number of normal or attacks samples shouldn't be bigger than 4000`
    """
    # ---- Dataset loading 
    training = np.load("data/training_normal.npy")
    testing_normal = np.load("data/testing_normal.npy")
    testing_attack = np.load("data/testing_attack.npy")
    
    # ---- Creation of the testing set & labels
    # number of rows in normal/attack testing subdatasets
    n_testing_attack = int(n_testing_samples*attacks_ratio)
    n_testing_normal = n_testing_samples-n_testing_attack

    # exit if there isn't enough samples in normal or attack datasets
    if n_testing_attack>testing_attack.shape[0] or n_testing_normal>testing_normal.shape[0]:
        print("Error: not enough samples in testing set")
        exit()
    
    # select random rows in each dataset: create subdatasets based on ratio
    testing_normal_index = np.random.choice(testing_normal.shape[0], size = n_testing_normal, replace = False)
    testing_attack_index = np.random.choice(testing_attack.shape[0], size = n_testing_attack, replace = False)
    
    # creation of the labels
    labels = np.concatenate(
        [
            np.full(n_testing_normal, 0),
            np.full(n_testing_attack, 1)
        ], axis = 0
    )

    # creation of the labels: merge of normal and attack subdatasets
    testing = np.concatenate(
        [
            testing_normal[testing_normal_index,:],
            testing_attack[testing_attack_index,:]
        ], axis = 0
    ) 
    
    return training , testing, labels


def score(predicted_labels, actual_labels, metrics=["confusion_mat"]):
    """
    ## Compute and print score metrics between predicted and actual labels

        `parameters`
            predicted_labels: predicted labels
            actual_labels: actual labels
            metrics: array of metrics to compute (must be included in ["confusion_mat", "accuracy", "TPR", "recall", "FPR", "F1score"])
        `return`
            scores: score metric if return_value is set to True
    """
    n = len(predicted_labels) # lengths of the arrays
    scores = {}

    # Compute confusion matrix rates
    TN= TP= FP= FN= 0
    for i in range(n):
        if predicted_labels[i] == actual_labels[i]:
            if predicted_labels[i]==1:
                TP +=1
            else:
                TN +=1
        else:
            if predicted_labels[i]==1:
                FP +=1
            else:
                FN +=1

    # ---- Compute and print metrics
    beautify = lambda x: str(np.round(x*100, 3)) + " %"
    
    if "confusion_mat" in metrics: # Confusion matrix
        print("Confusion matrix:" )
        print(f"       | {'normal':<8} | {'attack':<8}")
        print(f"normal | {beautify(TN/n):>8} | {beautify(FN/n):>8}")
        print(f"attack | {beautify(FP/n):>8} | {beautify(TP/n):>8}")

    if "accuracy" in metrics: # Accuracy
        accuracy= (TP+TN)/(TP+FP+TN+FN)
        scores['accuracy'] = accuracy
        print("accuracy=", beautify(accuracy)) 

    precision= TP/(TP+FP) # Precision
    if "precision" in metrics:
        scores['precision'] = precision
        print("precision=", beautify(precision)) 
    
    TPR= TP/(TP+FN) # True Positive Rate or recall
    if "TPR" in metrics:
        scores['TPR'] = TPR
        print("TPR=", beautify(TPR))

    FPR= FP/(FP+TN) # False Positive Rate or fall-out
    if "FPR" in metrics:
        scores['FPR'] = FPR
        print("FPR=", beautify(FPR)) 

    if "F1score" in metrics: # F1-Score
        F1score=(2*precision*TPR)/(precision+TPR)
        scores['F1score'] = F1score
        print("F1-score=",F1score)

    return(scores)


def epsilon_start():
    """
    Function that plots the sorted distances of the nearest neighbor for all the samples of the PCA projected training set
    the point with the greatest curvature is a decent starting value for epsilon in the clustering algorithms
    """
    training, testing, _ = get_data(n_testing_samples=4000, attacks_ratio = .2)
    condensed_training, _ = PCA.reduce_dimensions(training, testing, 10)

    # Compute the distances with all the points
    distances = sklearn.metrics.pairwise.euclidean_distances(condensed_training,condensed_training)

    distances = np.sort(distances, axis = 0) # sort the distances (shortest distance will be in the row 1)
    distances = np.sort(distances[1,:]) # we keep only the distance with the nearest neighbor and sort this array

    # Visualization
    plt.plot(distances)
    plt.xlabel("sorted samples"); plt.ylabel("distance of the nearest neighbor")
    plt.show()


# ------------- TEST OF THE FUNCTIONS
if __name__ == '__main__':
    # Load data
    get_data(n_testing_samples=4000, attacks_ratio = .2)
    
    # Test score function
    score([0, 1, 0, 1], [0, 0, 1, 1], metrics=['accuracy'])

    # Plot starting value for epsilon
    epsilon_start()
