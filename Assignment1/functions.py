# Modules import
import re           # regular expression
import random       # to split dataset randomly
import numpy as np  # for round and logarithm functions

"""
Tokenizes all raw messages of a text file
    parameters
        filename: name of the textfile
    returns:
        tokens: arrays of tokens for each message
"""
def tokenize(filename):
    with open(filename) as file:
        lines=file.readlines()
        tokens=[]
        for i in range(0, len(lines)):
            line = lines[i]
            words = re.findall("[\w'-]+", re.sub("\t", " ", line).lower()) # array of words and numbers (apostrophes are kept)
            ponctuations = re.findall("[^A-Za-z0-9\s]", re.sub("\t", " ", line)) # array of ponctuations 
            
            tokens.append([*words, *ponctuations]) # we can concatenate these arrays because the order of words isn't used

    return tokens

"""
Randomly split data into two complementary sets with sizes based on ratio
    parameters:
        tokens: [array] data to split 
        ratio: [float between 0 and 1] balance of the size of the two sets
    returns:
        training_set: bigger set of tokens
        test_set: smaller set of tokens (complementary to training set)
"""
def split_data(tokens, ratio = .7):
    training_ind = random.sample(range(len(tokens)), int(len(tokens)*ratio))
    training_set = [tokens[i] for i in training_ind]
    test_set = [tokens[i] for i in range(len(tokens)) if i not in training_ind]
    return training_set, test_set

"""
Print the balance between categories in a dataset 
    parameters:
        tokens: arrays of tokens for each message
        title: title of the print in the console
"""
def category_balance(tokens, title = 'Category balance'):
    labels = [m[0] for m in tokens]
    categories,values = np.unique(labels, return_counts=True)
    print(title, ":\t", dict(zip(categories,values)), ", ratio =", str(100*np.round(values[1]/values[0], 3)) + "%")


"""
Score function which prints metrics
    parameters:
        p_label: predicted labels
        t_label: true labels
        metrics: array of metrics to compute (must be included in ["confusion_mat", "accuracy", "precision", "recall", "F1score"])
        return_value: if False print the metric, else return it (False by default) 
    returns:
        score: [optional] score metric if return_value is set to True
"""
def score(p_label, t_label, metrics=["confusion_mat"], return_value=False):
    n = len(p_label) # size of the test set
    
    # Compute confusion matrix rates
    TN= TP= FP= FN= 0
    for i in range(n):
        if p_label[i] == t_label[i]:
            if p_label[i]=='spam':
                TP +=1
            else:
                TN +=1
        else:
            if p_label[i]=='spam':
                FP +=1
            else:
                FN +=1

    beautify = lambda x: str(np.round(x*100, 3)) + " %"
    # Compute and print metrics
    if "confusion_mat" in metrics:
        print("Confusion matrix:" )
        print(f"     | {'ham':<8} | {'spam':<8}")
        print(f"ham  | {beautify(TN/n):>8} | {beautify(FN/n):>8}")
        print(f"spam | {beautify(FP/n):>8} | {beautify(TP/n):>8}")

    if "accuracy" in metrics:
        accuracy= (TP+TN)/(TP+FP+TN+FN)
        if return_value: 
            return accuracy
        else:
            print("accuracy=",beautify(accuracy)) 

    precision= TP/(TP+FP)
    if "precision" in metrics:
        if return_value: 
            return precision
        else:
            print("precision=", beautify(precision)) 

    recall= TP/(TP+FN)
    if "recall" in metrics:
        if return_value: 
            return recall
        else:
            print("recall=", beautify(recall))
    
    if "F1score" in metrics:
        F1score=(2*precision*recall)/(precision+recall)
        if return_value: 
            return F1score
        else:
            print("F1-score=",F1score)

