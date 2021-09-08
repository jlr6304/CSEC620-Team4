import sys
import re
import random
import kNN
import NaiveBayes


def tokenize(filename):
    with open(filename) as file:
        lines=file.readlines()
        tokens=[]
        for i in range(0, len(lines)):
            tokens.append(re.findall("[\w'-]+", re.sub("\t", " ", lines[i]).lower()))
    return tokens


def split_data(tokens):
    ratio = .7
    training_ind = random.sample(range(len(tokens)), int(len(tokens)*ratio))
    training_set = [tokens[i] for i in training_ind]
    test_set = [tokens[i] for i in range(len(tokens)) if i not in training_ind]
    return training_set, test_set


def main():
    tokens = tokenize(sys.argv[1])
    training_set, test_set = split_data(tokens)
    kNN.run(training_set, test_set)
    NaiveBayes.run(training_set, test_set)


if __name__ == '__main__':
    main()
