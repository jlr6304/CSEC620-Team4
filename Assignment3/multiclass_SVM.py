import itertools
import pandas as pd
import time
from Assignment3.functions import accuracy
from SVM import *


def confusion_matrix_malware(pred_labels, true_labels):
    """
    #### Print the confusion matrix from the given labels (can be multicategorical)
    """
    n = len(pred_labels)

    values, counts = np.unique(np.concatenate((true_labels, pred_labels)), return_counts=True)

    ind = np.argpartition(-counts, kth=20)[:20]
    levels = list(values[ind])  # prints the 20 most frequent elements

    index = dict(zip(levels, range(len(levels))))
    conf_mat = pd.DataFrame(0, index=levels, columns=levels)

    for i,j in zip(true_labels, pred_labels):
        if i in levels and j in levels:
            conf_mat.loc[i, j] += 1

    # Print
    (conf_mat/n).to_csv('temp.csv', sep='\t')
    print(bold("Confusion matrix:"))
    print(conf_mat/n)
    print("levels:", levels, '\n')


def split_train_test_malware(df, labels):
    """
    #### Split the `df` dataset into training/testing samples based on the 70/30 train/test split.

    Returns the train and test dataset with the associated labels
    """
    split_mask = np.random.choice([True, False], len(labels), p=[0.3, 0.7])
    train_set = df[np.logical_not(split_mask), :]
    test_set = df[split_mask, :]

    train_labels = labels[np.logical_not(split_mask)]
    test_labels = labels[split_mask]

    return train_set, train_labels, test_set, test_labels


def to_binary_categories_malware(labels, classify_func):
    """
    #### Convert nominal categories into -1, 1 categories based on the `classify_func`
    return the new labels
    """
    return np.vectorize(classify_func)(labels)


def to_cleartext_categories_malware(labels, recreate_func):
    """
    #### Convert binary categories into cleartext categories based on the `classify_func`
    return the new labels
    """
    return np.vectorize(recreate_func)(labels)


def distance_point_hyperplane(x, hyperplane):
    """
    find the distance between a nD point and a hyperplane
    :param x: an nD point
    :param hyperplane: a hyperplane defined by a slope and intercept
    :return: the distance between the point and the hyperplane
    """
    w = hyperplane['slope']
    b = hyperplane['intercept']
    d = np.abs(np.dot(w, x)-b)/np.linalg.norm(w, 2)
    return d.item()


def one_vs_all(train_set, train_labels, test_set, test_labels):
    """
    - Train one classifier for each of your classes.
    - Each classifier is trained to identify if a sample belongs to either the target class or one of the other classes
    (labeled as +1 for target class, -1 for any other class).
    - To perform multi-class classification, use each model to predict the sample. Use the target class for whichever
    model has the boundary farthest from the sample (i.e. the y prediction that is the highest positive value).

    :return: None
    """
    family_data = {}
    unique_labels_malware = set(list(train_labels))
    # generate multiclass hyperparameters
    count = 0
    for target_class in unique_labels_malware:
        print(count/len(unique_labels_malware)*100, "% complete")
        count += 1
        binary_labels = to_binary_categories_malware(train_labels, lambda x: 1 if x == target_class else -1)
        hyperplane, l = fit(train_set, binary_labels)
        family_data[target_class] = hyperplane

    # predict samples based on max distance from hyperplane.
    pred_labels = np.empty(0)
    for sample in test_set:
        # calculate distances from every generated hyperparameter
        prediction_count = dict(zip(set(list(train_labels)), np.zeros(len(set(list(train_labels))))))
        for family in family_data:
            prediction = predict(np.array([sample]), family_data[family])
            prediction_count[family] += prediction[0]
        max_key = max(prediction_count, key=prediction_count.get)
        max_values = []
        max_keys = []
        # find families and hyperplanes with highest count.
        for key in prediction_count:
            if prediction_count[key] == prediction_count[max_key]:
                max_values.append(family_data[key])
                max_keys.append(key)
        distances = [distance_point_hyperplane(sample, hyperplane) for hyperplane in max_values]
        # use distances from hyperplane to break ties
        family = max_keys[distances.index(max(distances))]
        # add family furthest away from hyperplane to predicted labels
        pred_labels = np.append(pred_labels, family)

    # show confusion matrix
    confusion_matrix_malware(pred_labels, test_labels)
    # show accuracy
    accuracy(pred_labels, test_labels, verbose=True)


def one_vs_one(train_set, train_labels, test_set, test_labels):
    """
    - Train classes * (classes-1)2 number of classifiers.
    - Each classifier is trained to identify if a sample belongs to one of two particular classes in the multi-class
    dataset (e.g. spyware or cryptoware).
    - To perform multi-class classification, use each model to predict the sample. Use voting to determine which class
    the sample belongs to.

    :return: none
    """
    # generate possible classification combinations
    combinations = set(itertools.combinations(set(train_labels), 2))
    count = 0
    # create voting struct
    ballot = dict(zip(np.arange(len(list(test_labels))), list(train_labels)))
    for i in ballot:
        ballot[i] = dict(zip(set(list(train_labels)), np.zeros(len(set(train_labels)))))

    # train one classifier for each possible combination
    for combination in combinations:
        print((count/len(combinations))*100, " % complete")
        count += 1
        print(f"training model for {combination[0]} and {combination[1]}")

        binary_train_labels = np.array([label for label in train_labels if label in combination])
        binary_train_set = np.array([train_set[i] for i in range(0, len(train_labels)) if train_labels[i] in combination])

        binary_train_labels = to_binary_categories(binary_train_labels, lambda x: 1 if x == combination[0] else -1)
        hyperplane, l = fit(binary_train_set, binary_train_labels)
        # predict all samples using trained classifier
        for i in range(0, len(test_labels)):
            prediction = predict(np.array([test_set[i]]), hyperplane)
            cleartext_prediction = to_cleartext_categories_malware(prediction, lambda x: combination[0] if x == 1 else combination[1])
            # vote for one of two predicted samples
            ballot[i][cleartext_prediction[0]] += 1

    pred_labels = np.empty(0)
    # find sample with largest number of votes
    for index in ballot:
        max_key = max(ballot[index], key=ballot[index].get)
        pred_labels = np.append(pred_labels, max_key)

    # show confusion matrix
    confusion_matrix_malware(pred_labels, test_labels)
    # show accuracy
    accuracy(pred_labels, test_labels, verbose=True)



def main():
    """

    :return: none
    """
    df, labels = import_data()
    labels_malware = np.delete(labels, (np.where(labels == 'Benign')))
    df_malware = np.delete(df, (np.where(labels == 'Benign')), axis=0)

    train_set, train_labels, test_set, test_labels = split_train_test_malware(df_malware, labels_malware)

    start = time.time()
    one_vs_all(train_set, train_labels, test_set, test_labels)
    end = time.time()
    print(f"one_vs_all execution time: {end-start}")

    start = time.time()
    one_vs_one(train_set, train_labels, test_set, test_labels)
    end = time.time()
    print(f"one_vs_one execution time: {end-start}")


if __name__ == "__main__":
    main()
