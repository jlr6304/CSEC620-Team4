import pandas as pd
import numpy as np


def to_binary_categories(labels, classify_func = lambda x: -1 if x=='Benign' else 1):
    """
    #### Convert nominal categories into -1, 1 categories based on the `classify_func`
    return the new labels
    """
    return np.vectorize(classify_func)(labels)
    
def recreate_categories(labels, recreate_func = lambda x: 'Benign' if x==-1 else 'Malware'):
    """
    #### Recreate the nominal categories from a {-1, 1}-array categories based on the `recreate_func`
    return the new labels
    """
    return np.vectorize(recreate_func)(labels)

def split_train_test(df, labels, features, total_samples=4000, test_ratio=.3, malware_ratio=.2):
    """
    #### Split the `df` dataset into training/testing samples based on the ratios
    `Note:` it keeps the ratio of the malwares families of the source dataset

    Returns the train and test dataset with the associated labels
    """

    split_ratio = total_samples/len(df)

    fams = np.unique(labels, return_counts=True)
    fams = dict(zip(fams[0], fams[1]))
    
    index_train = set()
    index_test = set()
    
    for f in fams.keys():
        if f=="Benign":
            number_samples = total_samples*(1-malware_ratio)
        else:
            number_samples = total_samples*malware_ratio*fams[f]/(len(df)-fams['Benign'])
            
        # number_samples =  if f == "Benign" else split_ratio*malware_ratio

        index = np.where(labels==f)[0]
        # print('Family', f, ':', fams[f]*ratio)
        
        index = np.random.choice(index, int(number_samples), replace=False)
        
        # split train/test
        index_train_fam = np.random.choice(index, int(len(index)*(1-test_ratio)), replace=False)
        index_test_fam = list(set(index)-set(index_train_fam))

        index_train = index_train.union(set(index_train_fam))
        index_test = index_test.union(set(index_test_fam))

    # print(np.unique(labels[list(index_train)], return_counts=True))
    # print(np.unique(labels[list(index_test)], return_counts=True))
    
    train_labels = labels[list(index_train)]
    test_labels = labels[list(index_test)]

    train_set = df[list(index_train), :]
    test_set = df[list(index_test), :]
    
    # Remove empty features in the train data
    empty_features = np.where(np.apply_along_axis(sum, 0, train_set)<=0)
    train_set = np.delete(train_set, empty_features, axis=1)
    test_set = np.delete(test_set, empty_features, axis=1)
    features = np.delete(features, empty_features[0], axis=0)

    # print(features.shape)
    # print(train_set.shape)
    # print(test_set.shape)
    
    return train_set, train_labels, test_set, test_labels, features

def import_data():
    df = np.load('data/samples.npy', allow_pickle=True)
    labels = np.load('data/family.npy', allow_pickle=True)
    features = np.load('data/features.npy', allow_pickle=True)
    return df, labels, features

def bold(string):
    return '\033[1m' + string + '\033[0m'

def confusion_matrix(pred_labels, true_labels):
    """
    #### Print the confusion matrix from the given labels (can be multicategorical)
    """
    n = len(pred_labels)
    levels = list(np.unique(np.concatenate((true_labels, pred_labels), axis=0)))    
    
    index = dict(zip(levels, range(len(levels))))
    conf_mat = pd.DataFrame(0, index=levels, columns=levels)
    
    for i,j in zip(true_labels, pred_labels):
        conf_mat.loc[i,j] += 1

    # Print
    print(bold("Confusion matrix:"))
    print(conf_mat/n)
    print("levels:", levels, '\n')

def f1_score(pred_labels, true_labels, verbose=False):
    """
    #### Compute the F1-score from the given labels (printing the result is optionnal)
    """

    TP = sum((pred_labels == 'Malware') & (true_labels == 'Malware'))
    FP = sum((pred_labels == 'Malware') & (true_labels == 'Benign'))
    FN = sum((pred_labels == 'Benign') & (true_labels == 'Malware'))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    f_score = 2*precision*recall/(precision+recall)

    if verbose:
        print(bold("F1-Score:"), np.round(f_score, 4), '\n')
    else:
        return f_score

def accuracy(pred_labels, true_labels, verbose=False):
    """
    #### Compute the accuracy of the given labels (printing the result is optionnal)
    """
    acc = (pred_labels == true_labels).mean()
    
    if verbose:
        print(bold("Accuracy:"), np.round(100*acc, 2), '%\n')
    else:
        return acc

def distance_point_hyperplane(x, hyperplane):
    w = hyperplane['slope']
    b = hyperplane['intercept']
    d = np.abs(np.dot(w, x)-b)/np.linalg.norm(w, 2)
    return d.item()

if __name__=='__main__':

    # ----- Test of the data import and the train/test split
    # df, labels = import_data()

    # train_set, train_labels, test_set, test_labels = split_train_test(df, labels)

    # train_labels = to_binary_categories(train_labels)
    # test_labels = to_binary_categories(test_labels)

    # print(np.unique(train_labels, return_counts=True))
    # print(np.unique(test_labels, return_counts=True))
    
    # ----- Test of the confusion matrix function
    t = np.array(['b', 'm', 'c', 'a'])
    p = np.array(['b', 'b', 'm', 'm'])
    confusion_matrix(p, t)

