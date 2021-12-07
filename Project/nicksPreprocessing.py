import math
import numpy as np
from google_images_download import google_images_download
import time
import csv
import pandas as pd

def get_bad_features():
    with open("bad_features.txt") as f:
        all_lines = f.read()
        lines = all_lines.splitlines()
        return lines

def remove_bad_features():
    bad_features=get_bad_features()
    unique_bad_features=[]
    for x in bad_features:
        if x not in unique_bad_features:
            unique_bad_features.append(x)
    data = pd.read_csv('metadata.csv', encoding='latin-1', low_memory=False)
    print(data.axes)
    Malicious_array=[]
    directories=data['SourceFile']
    for path in directories:
        path=path.lower()
        if "malware" in path:
            Malicious_array.append("1")
        if "clean" in path:
            Malicious_array.append("0")
        if "malware" not in path and "clean" not in path:
            Malicious_array.append("idk")
    data['Malicious'] = Malicious_array
    for bad_feature in unique_bad_features:
        print(bad_feature)
        data.drop(bad_feature, inplace=True, axis=1)
        print(data)
    data.to_csv("new_metadata.csv",index=False)
    print("done")


def find_all_unique_values_fetures(data):
    all_same_fetures = []
    for column in data:
        print(column)
        a = data[column].to_numpy().astype(str)
        if (np.unique(a).size == len(a)) == True:
            all_same_fetures.append(column)
            print(column, " has all unique values")
    return all_same_fetures
def find_all_same_fetures(data):
    all_same_fetures=[]
    for column in data:
        a = data[column].to_numpy()
        if ((a[0] == a).all()) == True:
            all_same_fetures.append(column)
    return all_same_fetures
def find_with_few_value_fetures(data,num):
    fetures = []
    for column in data:
        a = data[column].to_numpy()
        bad = True
        notempies = []
        one_value = a[2]
        for value in a:
            if value == value and value != " ":
                bad = False
                notempies.append(value)
        if len(notempies) < num:
            fetures.append(column)
    return fetures
def find_empty_fetures(data):
    empty_fetures=[]
    for column in data:
        a = data[column].to_numpy()
        bad = True
        for value in a:
            if value == value and value !=" ":
                bad = False
        if bad == True:
            empty_fetures.append(column)
            print("---------all empty--------------")
            print(column)
            print(a[2])
            print(" ")
            print(" ")
    return empty_fetures
def add_to_bad_fetures(columns):
    file = open("bad_features.txt", 'r')
    found_bad_feautes = file.read().splitlines()
    bad_features = open("bad_features.txt", "a")
    for column in columns:
        if column not in found_bad_feautes:
            bad_features.write(column + "\n")
def find_bad_features():
    data = pd.read_csv('metadata.csv', encoding='latin-1', low_memory=False)
    empty_fetures=find_empty_fetures(data)
    unique_values=find_all_unique_values_fetures(data)
    all_same_fetures=find_all_same_fetures(data)
    few_values=find_with_few_value_fetures(data,3)
    add_to_bad_fetures(empty_fetures)
    add_to_bad_fetures(unique_values)
    add_to_bad_fetures(all_same_fetures)
    add_to_bad_fetures(few_values)

find_bad_features()
remove_bad_features()
