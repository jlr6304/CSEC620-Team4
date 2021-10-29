import time
from Assignment3.functions import accuracy
from SVM import *

def split_train_test_malware(df, labels):
    """
    #### Split the `df` dataset into training/testing samples based on the 70/30 train/test split.

    Returns the train and test dataset with the associated labels
    """

    # print(df.shape)
    # print(labels.shape)
    split_mask = np.random.choice([True, False], len(labels), p=[0.3, 0.7])
    train_set = df[np.logical_not(split_mask), :]
    test_set = df[split_mask, :]

    # print(train_set.shape)
    # print(test_set.shape)

    train_labels = labels[np.logical_not(split_mask)]
    test_labels = labels[split_mask]

    # print(train_labels.shape)
    # print(test_labels.shape)

    return train_set, train_labels, test_set, test_labels


def to_binary_categories_malware(labels, classify_func):
    """
    #### Convert nominal categories into -1, 1 categories based on the `classify_func`
    return the new labels
    """
    return np.vectorize(classify_func)(labels)


def to_cleartext_categories_malware(labels, recreate_func):
    """
    #### Convert nominal categories into -1, 1 categories based on the `classify_func`
    return the new labels
    """
    return np.vectorize(recreate_func)(labels)


def distance_point_hyperplane(x, hyperplane):
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
        # print(train_labels)
        binary_labels = to_binary_categories_malware(train_labels, lambda x: 1 if x == target_class else -1)
        # print(binary_labels)
        hyperplane, l = fit(train_set, binary_labels)
        family_data[target_class] = hyperplane

    # predict samples based on max distance from hyperplane.
    pred_labels = np.empty(0)
    for sample in test_set:
        # print(sample[1])
        # calculate distances from every generated hyperparameter
        prediction_count = dict(zip(set(list(train_labels)), np.zeros(len(set(list(train_labels))))))
        for family in family_data:
            prediction = predict(np.array([sample]), family_data[family])
            prediction_count[family] += prediction[0]
        max_key = max(prediction_count, key=prediction_count.get)
        # distances = [distance_point_hyperplane(sample, hyperplane) for hyperplane in list(family_data.values())]
        # deduce classifier single family from distances
        # family = list(family_data.keys())[np.argmax(distances)]
        # add prediction based on that family to predicted labels
        pred_labels = np.append(pred_labels, max_key) # to_cleartext_categories_malware(predict(np.array([sample]), family_data[family]), lambda x: family if x == 1 else "Other"))

    # print(pred_labels)
    # print(test_labels)
    # show confusion matrix
    confusion_matrix(pred_labels, test_labels)
    # show accuracy
    accuracy(pred_labels, test_labels, verbose=True)
    return



def one_vs_one():
    """
    - Train classes * (classes-1)2 number of classifiers.
    - Each classifier is trained to identify if a sample belongs to one of two particular classes in the multi-class
    dataset (e.g. spyware or cryptoware).
    - To perform multi-class classification, use each model to predict the sample. Use voting to determine which class
    the sample belongs to.

    :return:
    """
    pass

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
    print(f"execution time: {end-start}")



if __name__ == "__main__":
    main()
