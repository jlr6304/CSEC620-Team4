# Modules import
from time import time # to measure execution times 
from functions import tokenize, split_data, score, category_balance

# Created classifiers
import kNN
import NaiveBayes

"""
Compares kNN and Mutlinomial Naive Bayes classifiers on the SMS Spam Collection dataset 
"""
def main():
    filename = "SMSSpamCollection"
    
    # Tokenize data
    tokens = tokenize(filename)
    training_set, test_set = split_data(tokens, ratio = .7) # split into train and test set

    # kNN classifier (parameters described in kNN.py)
    print("kNN classifier\n" + '-'*20)
    t = time()
    predicted_labels_kNN, true_labels = kNN.run(
        training_set, 
        test_set,
        k_param = 5,
        distances = "euclidean",
        feature_select_threshold = 2,
        balance_threshold = .3
    )
    t_kNN = time()-t #time for kNN

    # Naive Bayes classifier
    print("Naive Bayes classifier\n" + '-'*20)
    t = time()
    predicted_labels_NB, true_labels = NaiveBayes.run(training_set, test_set)
    t_NB = time()-t #time for Naive Bayes

    # Execution time comparison 
    print('\n' + '\033[1m' + "Execution time comparison" + '\033[0m')
    print(f"time kNN: {t_kNN}, \t time Naive Bayes: {t_NB}")
    
    # Score comparison
    print('\033[1m' + "kNN classifier" + '\033[0m') # kNN scores
    score(predicted_labels_kNN, true_labels, metrics = ['confusion_mat', 'accuracy', 'precision', 'recall', 'F1score'])

    print('\033[1m' + "Naive Bayes classifier" + '\033[0m') # Naive Bayes scores
    score(predicted_labels_NB, true_labels, metrics = ['confusion_mat', 'accuracy', 'precision', 'recall', 'F1score'])

if __name__ == '__main__':
    main()
