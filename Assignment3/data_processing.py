import os
import time
import datetime
import csv
import random
def get_malware_samples():
    malware_samples_with_group=[]
    malware_samples=[]
    with open('C:\\Users\\nickm\\Downloads\\sha256_family.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            malware_samples_with_group.append(row[0].split(","))
            malware_samples.append(row[0].split(",")[0])
    return malware_samples_with_group,malware_samples

def get_malware_and_benign_samples_and_features(malware_names,ratio_of_malware,number_of_samples_to_collect):
    benign_samples=[]
    malware_samples=[]
    number_of_malware_samples_to_collect = (ratio_of_malware * number_of_samples_to_collect)
    number_of_benign_samples_to_collect=((1-ratio_of_malware)*number_of_samples_to_collect)
    number_of_samples_collected=0
    files = os.listdir("C:\\Users\\nickm\Downloads\\feature_vectors\\feature_vectors")
    number_of_files=len(files)
    files_used=[]
    # go through each file in the feature_vectors directory and extract the contents of the file
    while number_of_samples_collected < number_of_samples_to_collect:
        file_position=random.randrange(number_of_files)
        if file_position not in files_used:
            files_used.append(file_position)
            file = files[file_position]
            with open("C:\\Users\\nickm\Downloads\\feature_vectors\\feature_vectors\\" + file) as f:
                lines = f.readlines()
                feature_vectors_sample = (file, lines)
                print(" ")
                print("number_of_samples_to_collect: ", number_of_samples_to_collect)
                print("number_of_beign_samples_to_collect: ", number_of_benign_samples_to_collect)
                print("number_of_malware_samples_to_collect: ", number_of_malware_samples_to_collect)
                print("benign samples collected: ", len(benign_samples))
                print("malware samples collected: ", len(malware_samples))
                # if the name of the sample is the name of a malware smaple, then add it to the list of malware samples, otherwise add it to the benign samples
                if file in malware_names:
                    if (len(malware_samples) < number_of_malware_samples_to_collect ):
                        malware_samples.append(feature_vectors_sample)
                        number_of_samples_collected+=1
                else:
                    if(len(benign_samples)<number_of_benign_samples_to_collect):
                        benign_samples.append(feature_vectors_sample)
                        number_of_samples_collected += 1
    return malware_samples,benign_samples
def get_unique_features(malware_samples, benign_samples):
    unique_features = []
    for sample in malware_samples:
        (sample_name, feature_vectors) = sample
        # print(sample_name) to print the name of the current sample
        # print(feature_vectors) to print the current features
        for vector in feature_vectors:
            # example: vector=intent::android.intent.action.MAIN
            if vector not in unique_features:
                unique_features.append(vector)
    for sample in benign_samples:
        (sample_name, feature_vectors) = sample
        # print(sample_name) to print the name of the current sample
        # print(feature_vectors) to print the current features
        for vector in feature_vectors:
            # example: vector=intent::android.intent.action.MAIN
            if vector not in unique_features:
                unique_features.append(vector)
    return unique_features
def get_feature_vectors(samples, unique_features):
    samples_and_feature_vectors=[]
    for sample in samples:
        feature_vector = []
        (sample_name, feature_vectors) = sample
        for feature in unique_features:
            if feature in feature_vectors:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        sample_name_feature_vector = (sample_name, feature_vector)
        # example: sample_name_feature_vector=('13539e5b4d9ee32b5390aef4668a7e0f83a070775f909c2a4c8f63747aeff865', [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        samples_and_feature_vectors.append(sample_name_feature_vector)
    return samples_and_feature_vectors

start = time.time()
now = datetime.datetime.now()
print("start:",now)
ratio_of_malware=0.6 #the percent of samples collected that will be malware
number_of_samples_to_collect= 1000 #total number of samples to collect
malware_samples_with_group,malware_names=get_malware_samples() #get a list of malware names with their groups, and a list with just the names

malware_samples,benign_samples=get_malware_and_benign_samples_and_features(malware_names,ratio_of_malware,number_of_samples_to_collect)
unique_features=get_unique_features(malware_samples, benign_samples) #create a list of unique features
malware_names_and_vectors=get_feature_vectors(malware_samples, unique_features) #get a list of the malware samples' feature vectors with their names
benign_names_and_vectors=get_feature_vectors(benign_samples, unique_features)  #get a list of the benign samples' feature vectors with their names
print(malware_samples)
for i in malware_names_and_vectors:
    print(i)
end = time.time()
now = datetime.datetime.now()
print("getting feature vectors:", end - start)
print("end:",now)
