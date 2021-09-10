import sys
import re
import random
# import kNN
import NaiveBayes

import numpy as np

def tokenize(filename):
    with open(filename) as file:
        lines=file.readlines()
        tokens=[]
        for i in range(0, len(lines)):
            line = lines[i]

            words = re.findall("[\w'-]+", re.sub("\t", " ", line).lower()) # => array of words / numbers
            ponctuations = re.findall("[^A-Za-z0-9\s]", re.sub("\t", " ", line)) # => array of ponctuations 
            
            tokens.append([*words, *ponctuations])

    return tokens

def split_data(tokens, ratio = .7):
    training_ind = random.sample(range(len(tokens)), int(len(tokens)*ratio))
    training_set = [tokens[i] for i in training_ind]
    test_set = [tokens[i] for i in range(len(tokens)) if i not in training_ind]
    return training_set, test_set


"""
Score function which prints metrics
    p_label : predicted labels
    t_label : true labels
    metrics : array of metrics to compute & print
"""
def score(p_label, t_label, metrics=["confusion_mat"]):
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

        print(beautify(TN/n), '\t', beautify(FN/n))
        print(beautify(FP/n), '\t', beautify(TP/n))

    if "accuracy" in metrics:
        accuracy= (TP+TN)/(TP+FP+TN+FN)
        print("accuracy=",beautify(accuracy)) 

    precision= TP/(TP+FP)
    if "precision" in metrics:
        print("precision=", beautify(precision)) 

    recall= TP/(TP+FN)
    if "recall" in metrics:
        print("recall=", beautify(recall)) 
    
    if "F1score" in metrics:
        F1score=(2*precision*recall)/(precision+recall)
        print("F1-score=",F1score) 

def main():
    tokens = tokenize(sys.argv[1])
    training_set, test_set = split_data(tokens)
    # kNN.run(training_set, test_set)
    predicted_l, true_l = NaiveBayes.run(training_set, test_set)

    score(predicted_l, true_l, metrics = ['confusion_mat', 'accuracy', 'precision', 'recall', 'F1score'])
 

if __name__ == '__main__':
    main()