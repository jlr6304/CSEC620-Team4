from sklearn.svm import SVC
from time import time

from functions import import_data, split_train_test, to_binary_categories, recreate_categories, confusion_matrix, f1_score, accuracy, bold



def linearSVM(train_set, train_labels, test_set, C=1):
    """
    ### Process a linear SVM classifier using the `sklearn` package
        C corresponds to the regularization parameter
    """
    classifier = SVC(C=C, kernel='linear')
    classifier.fit(train_set, train_labels)

    pred_labels = classifier.predict(test_set)
    return pred_labels

def radialSVM(train_set, train_labels, test_set):
    """
    ### Process a radial SVM classifier using the `sklearn` package
    """
    classifier = SVC(kernel='rbf')
    classifier.fit(train_set, train_labels)

    pred_labels = classifier.predict(test_set)
    return pred_labels


if __name__ == '__main__':
    # import data from the preprocessed dataset
    df, labels, features = import_data()
    
    # split into training and testing sets
    train_set, train_labels, test_set, true_labels, features = split_train_test(
        df, labels, features, total_samples=4000, test_ratio=.3
    )

    # convert labels into appropriate format -1 and 1
    train_labels = to_binary_categories(train_labels, lambda x: x if x == 'Benign' else 'Malware')
    true_labels = to_binary_categories(true_labels, lambda x: x if x == 'Benign' else 'Malware')

    # --------- Process sklearn linear classifier 
    print('\n', bold('---------- SVM Linear classifier'), sep ='')
    # Training and classifying test samples
    s_t = time()
    pred_labels_linear = linearSVM(train_set, train_labels, test_set, C=.2)
    e_t = time()
    
    # Performances
    confusion_matrix(pred_labels_linear, true_labels) # Confusion matrix
    f1_score(pred_labels_linear, true_labels, verbose=True) # F1-score
    accuracy(pred_labels_linear, true_labels, verbose=True) # Accuracy
    print(bold("Execution time:"), e_t-s_t, 's\n')

    # --------- Process sklearn radial classifier
    print('\n', bold('---------- SVM Radial classifier'), sep ='')
    # Training and classifying test samples
    s_t = time()
    pred_labels_radial = radialSVM(train_set, train_labels, test_set)
    e_t = time()

    # Performances
    confusion_matrix(pred_labels_radial, true_labels) # Confusion matrix
    f1_score(pred_labels_radial, true_labels, verbose=True) # F1-score
    accuracy(pred_labels_radial, true_labels, verbose=True) # Accuracy
    print(bold("Execution time:"), e_t-s_t, 'seconds\n')
    