#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import pandas as pd
import tqdm

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import RandomForest
import time
import matplotlib.pyplot as plt

# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0

def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, #default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (port) samples.
    Xd_ts : numpy array
           Array containing testing (port) samples.
    Xc_tr : numpy array
           Array containing training (port) samples.
    Xc_ts : numpy array
           Array containing testing (port) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def randomforest_classification(X_tr, X_ts, Y_tr, Y_ts, labels, store_confusion=False):
    """
    Perform the classification procedure:
        train a random forest classifier using the NB prediction probabilities

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    store_confusion : bool
           If the confusion matrix need to be saved 
    """

    # -- Using skLearn (original version)
    # model = RandomForestClassifier(n_jobs=-1, n_estimators=1, oob_score=True)
    # model.fit(X_tr, Y_tr)
    # pred = model.predict(X_ts)

    # -- Using RandomForest.py module
    # fit the forest
    forest = RandomForest.fit(X_tr, Y_tr, n_trees=40, data_frac=.6, feature_subcount=4, max_depth = 10, min_node = 10)

    # predict the labels
    pred = RandomForest.predict(X_ts, forest, np.unique(Y_tr))

    # -- Performances
    # accuracy
    score = np.mean(pred==Y_ts) 
    print(f"RF accuracy = {score}")

    # classification report
    print(classification_report(Y_ts, pred, target_names=labels))

    Y_ts = labels[Y_ts]
    pred = labels[pred]
    
    # confusion matrix
    print(confusion_matrix(Y_ts, pred, labels=labels, normalize='true'))

    #   save confusion matrix
    if store_confusion:
        pd.DataFrame(
            confusion_matrix(Y_ts, pred, labels=labels, normalize='true'), columns=labels, index=labels
        )\
            .to_csv('confusion_matrix.csv', sep = ";")


def algorithm_comparison(X_tr, X_ts, Y_tr, Y_ts):
    """
    Compare the performances of Random Forest and Decision Tree classifiers:
        train a random forest classifier and a decision tree classifier and display training time and accuracy

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    """
    unique_Y = np.unique(Y_tr) # unique labels in training
    
    # -- Using Random Forest algorithm
    print("--- Random Forest training --- ")
    # fit the forest
    RF_start=time.time()
    forest = RandomForest.fit(X_tr, Y_tr, n_trees=40, data_frac=.6, feature_subcount=4, max_depth = 10, min_node = 10)
    RF_stop=time.time()

    # predict the labels
    RF_pred = RandomForest.predict(X_ts, forest, unique_Y)

    # Performance
    RF_score = np.mean(RF_pred==Y_ts) 
    RF_time=(RF_stop-RF_start)

    
    # -- Using Decision Tree algorithm
    print("--- Decision Tree training --- ")
    
    # fit the tree
    DT_start=time.time()
    tree = RandomForest.fit(X_tr, Y_tr, n_trees=1, data_frac=1, feature_subcount=X_tr.shape[1], max_depth = 10, min_node = 5)
    DT_stop=time.time()

    # predict the labels
    DT_pred = RandomForest.predict(X_ts, tree, unique_Y)

    # Performances
    DT_score = np.mean(DT_pred==Y_ts) 
    DT_time=(DT_stop-DT_start)
    

    # -- Performances comparison 
    print('\033[1m' + "Random Forest performances" + '\033[0m')
    print(f"RF accuracy = {RF_score}")
    print(f"RF time = {RF_time}s")

    print('\033[1m' + "Decision Tree performances" + '\033[0m')
    print(f"DT accuracy = {DT_score}")
    print(f"DT time = {DT_time}")


def node_importance(X_tr, X_ts, Y_tr, Y_ts):
    """
    Display the labels and the impurity of a node and its children in order to compute the node importance
        
    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    """
    unique_Y = np.unique(Y_tr) # unique labels of train set samples
    
    # -- Train Random Forest classifier
    print("--- Random Forest training --- ")
    # fit the forest
    forest = RandomForest.fit(X_tr, Y_tr, n_trees=20)

    # -- Select a node and show its labels and impurity
    parent = forest[0].right # can be changed

    print("parent labels: ", parent.labels)
    print("parent impurity: ", parent.impurity, end = 2*'\n')
    
    print("left child labels: \n", parent.left.labels)
    print("left child impurity: ", parent.left.impurity, end = 2*'\n')

    print("right child labels: \n", parent.right.labels)
    print("right child impurity: ", parent.right.impurity, end = 2*'\n')

    weighted_average = (parent.left.impurity*len(parent.left.labels) + parent.right.impurity*len(parent.right.labels)) / len(parent.labels)
    print("Node importance: ", parent.impurity - weighted_average)


def hyperparameter_tuning(X_tr, X_ts, Y_tr, Y_ts):
    """
    Method to tune the Random Forest classifier hyperparameters:
        Compute and graph the accuracy for a range of a chosen hyperparameter
        
    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    """
    # Hyperparameters list (single values are best hyperparameters) 
    n_trees_range = [30] #np.arange(1, 16)*5
    data_frac_range = [.2, .4, .6, .8, 1] # [.6]
    feature_subcount_range = [4] # [2, 3, 4, 5, 6]
    max_depth_range = [10] # [10, 20, 50, 100]
    min_node_range = [5] # [5, 10, 20, 50]

    score = []
    for n_trees in n_trees_range:
        for data_frac in data_frac_range:
            for feature_subcount in feature_subcount_range:
                for max_depth in max_depth_range:
                    for min_node in min_node_range:
                        
                        # Compute the accuracy of the corresponding forest
                        forest = RandomForest.fit(X_tr, Y_tr, \
                            n_trees,\
                            data_frac,\
                            feature_subcount,\
                            max_depth,\
                            min_node)    

                        pred = RandomForest.predict(X_ts, forest, np.unique(Y_tr))

                        score.append(np.mean(pred==Y_ts)) # add accuracy to the scores

    # Graph the results
    range = np.array(data_frac_range)
    score = np.array(score)

    plt.plot(range, score)
    plt.show()


def main(args):
    """
    Perform main logic of program
    """
    # load dataset
    print("Loading dataset ... ")
    X, X_p, X_d, X_c, Y = load_data(args.root)

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # # -------- Perform final classification
    print("Random Forest classification")
    randomforest_classification(X_tr_full, X_ts_full, Y_tr, Y_ts, le.classes_)

    # # -------- Decision Tree and Random Forest Comparison
    # print("DT and RF classifiers comparison")
    # algorithm_comparison(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # # ------ Node importance calculation
    # print("Node importance computation")
    # node_importance(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # # ------ Hyperparameter tuning
    # print("Hyperparameter tuning")
    # hyperparameter_tuning(X_tr_full, X_ts_full, Y_tr, Y_ts)

if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)
