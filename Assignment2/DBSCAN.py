import numpy as np
import sys
import time
import sklearn
from functions import get_data, score
import matplotlib.pyplot as plt

def find_points_in_epsilon(distances_of_points, epsilon, point_position):
    points_in_epsilon = []
    #for each point's distance to the point, if it is less than the epsilon distance, then add it to the points_in_epsilion list
    for new_point_position in range(0, len(distances_of_points)):
        if (distances_of_points[point_position][new_point_position] < epsilon):
            points_in_epsilon.append(new_point_position)
    return points_in_epsilon


def find_f1score(predicted_labels, actual_labels, metrics=["confusion_mat"]):
    n = len(predicted_labels)
    scores = {}
    #find the number of True negatives, true positives, False positives, and false negatives by seeing if labels match or not
    TN = TP = FP = FN = 0
    for i in range(n):
        if predicted_labels[i] == actual_labels[i]:
            #if the labels match and they are both 1, then it is a true positive, else it is a true negative
            if predicted_labels[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if predicted_labels[i] == 1:
                # if the labels do not match and the predicted label is 1 , then it is a false positive, else it is a false negative
                FP += 1
            else:
                FN += 1
    #the precison is calulated
    precision = TP / (TP + FP)
    # the ture positive rate is calculated
    TPR = TP / (TP + FN)
    # the flscore is calculated and added to the score list as F1score
    F1score = (2 * precision * TPR) / (precision + TPR)
    scores['F1score'] = F1score
    #return the score list
    return (scores)


def determine_core_points(distances_of_points, epsilon, min_neighbors, verbose=False):
    # print("\tdetermining core points in training set")
    core_points_positions = []
    # for each point in the distances_of_points list, find the number of points in epsilon
    for point_position in range(0, len(distances_of_points)):
        # if there is more or equal points in epsilon then the min_neighbors, then the point is a core point
        if len(find_points_in_epsilon(distances_of_points, epsilon, point_position)) >= min_neighbors:
            if verbose:
                print("core point found")
            core_points_positions.append(point_position)
    # return an list of positions of core points in the distances_of_points list
    return core_points_positions


def test(testing, training, core_points, epsilon, verbose=False):
    # print("\tanomaly detection on testing set")

    # find the distance of each test point to each and every core point
    distances_from_core_points = sklearn.metrics.pairwise.euclidean_distances(testing, core_points)
    labels = []
    # if the distance from any of the core points to the test point is less than the epsilon, then the point is not a anomaly, and the anomly boolean should be set to False, otherwise it will stay true
    for sample in distances_from_core_points:
        anomaly = True
        for distance in sample:
            if distance <= epsilon:
                anomaly = False
                break
        # a 1 should be added to the labels list if the anomaly boolean is true, otherwise add a 0 to the list
        labels.append(1 if anomaly else 0)

        # Print the category
        if verbose:
            if anomaly == True:
                print("anomaly")
            if anomaly == False:
                print("not anomaly")
    # return the list of 1 or 0s signifiying if a anomaly is present or not
    return labels

def get_predicted_labels(training, testing, min_neighbors, epsilon, verbose=False):
    # find the distances of each point to every other point
    distances_of_points = sklearn.metrics.pairwise.euclidean_distances(training, training)

    # find the positions of the core points in the distances_of_points list
    core_point_positions = determine_core_points(distances_of_points, epsilon, min_neighbors, verbose)
    core_points = []

    #for every core point found, put the details of the core point in the core_points list
    for position in core_point_positions:
        core_points.append(training[position])
    return test(testing, training, core_points, epsilon, verbose)

def generate_scores(epsilon,min_neighbors,predicted_labels,actual_labels):
    beautify = lambda x: str(np.round(x * 100, 3)) + " %"
    try:
        #find the f1score of the given labels
        metrics_list=find_f1score(predicted_labels, actual_labels, [ "F1score"])
        F1score = metrics_list["F1score"]
    except:
        F1score=0
        pass
    #print the values of the current iteration
    print(f"| {epsilon}  | {min_neighbors} | {F1score}")
    return F1score


def tune_parameters(training, testing, actual_labels,initial_min_neighbors, initial_epsilon, verbose=False):
    epsilon = initial_epsilon
    list_of_values=[]
    #set the number of iterations for the epsilon value to be put through
    epsilon_iterations=5
    # set the number of iterations for the min_neighbors value to be put through
    min_neighbors_iterations=5
    min_neighbors = initial_min_neighbors
    print(f"|eps       | k | F1-score")
    for epsilon_iteration in range(1,epsilon_iterations):
        lower_epsilon_f1scores = 0
        greater_epsilon_f1scores=0
        #seperate the lower and greater episoln values to make for cleaner data
        for min_neighbors_iteration in range(1,min_neighbors_iterations):
            #set the lower epsilon value to be the current epsilon value minus the value of the epsilon value divided by the epsilon_iteration value times 2, this will cause the eplison value to be half lower of what it
            lower_epsilon=epsilon-(epsilon/(epsilon_iteration*2))
            # set the lower min_neighbors value to be the current min_neighbors value minus the value of the min_neighbors value divided by the min_neighbors_iteration value times 2, this will cause the lower_min_neighbors value to be half lower of what it
            lower_min_neighbors = int(min_neighbors - (min_neighbors / (min_neighbors_iteration*2)))
            # get the predicted labels
            predicted_labels=get_predicted_labels(training, testing, lower_min_neighbors, lower_epsilon, verbose=False)
            # get the f1score
            lower_f1score=generate_scores(lower_epsilon,lower_min_neighbors,predicted_labels,actual_labels)
            # add the f1score to the total number of f1scores for all of this epsilon value
            lower_epsilon_f1scores=lower_epsilon_f1scores+lower_f1score
            # set the f1score for this min_neighbors
            lower_min_neighbors_f1score=lower_f1score
            #created the values tuple containing the epsilon, min_neighbors, predicted_labels, and f1score
            values = (lower_epsilon, lower_min_neighbors, predicted_labels, lower_f1score)
            #add the tuple to the list_of_values list
            list_of_values.append(values)
            #do the same as above for the min_neighbors value 1.5 as large
            greater_min_neighbors = int(min_neighbors + (min_neighbors /  (min_neighbors_iteration*2)))
            predicted_labels = get_predicted_labels(training, testing, greater_min_neighbors, lower_epsilon,verbose=False)
            lower_f1score = generate_scores(lower_epsilon, greater_min_neighbors, predicted_labels, actual_labels)
            lower_epsilon_f1scores = lower_epsilon_f1scores + lower_f1score
            greater_min_neighbors_f1score = lower_f1score
            values = (lower_epsilon, greater_min_neighbors, predicted_labels, lower_f1score)
            list_of_values.append(values)
            #if the f1score of the lower min_neighbors is greater than set the  min_neighbors as the lower_min_neighbors, otherwise set it as the greater_min_neighbors
            if lower_min_neighbors_f1score >= greater_min_neighbors_f1score:
                min_neighbors = lower_min_neighbors
            else:
                min_neighbors = greater_min_neighbors
        #after finding the best min_neighbors, reset it to the initial value
        min_neighbors =  initial_min_neighbors
        #do the same as above but for the epsilon 1.5 as large
        for min_neighbors_iteration in range(1,min_neighbors_iterations):
            greater_epsilon = epsilon + (epsilon/(epsilon_iteration*2))
            lower_min_neighbors = int(min_neighbors - (min_neighbors /  (min_neighbors_iteration*2)))
            predicted_labels = get_predicted_labels(training, testing, lower_min_neighbors, greater_epsilon, verbose=False)
            greater_f1score = generate_scores(greater_epsilon, lower_min_neighbors, predicted_labels,actual_labels)
            greater_epsilon_f1scores=greater_epsilon_f1scores+greater_f1score
            values=(greater_epsilon,lower_min_neighbors,predicted_labels,greater_f1score)
            lower_min_neighbors_f1score=greater_f1score
            list_of_values.append(values)
            greater_min_neighbors = int(min_neighbors + (min_neighbors /  (min_neighbors_iteration*2)))
            predicted_labels = get_predicted_labels(training, testing, greater_min_neighbors, greater_epsilon,verbose=False)
            greater_f1score = generate_scores(greater_epsilon, greater_min_neighbors, predicted_labels, actual_labels)
            greater_epsilon_f1scores = greater_epsilon_f1scores + greater_f1score
            values = (greater_epsilon, greater_min_neighbors, predicted_labels, greater_f1score)
            greater_min_neighbors_f1score = greater_f1score
            list_of_values.append(values)
            if lower_min_neighbors_f1score >= greater_min_neighbors_f1score :
                min_neighbors = lower_min_neighbors
            else:
                min_neighbors = greater_min_neighbors
        min_neighbors =  initial_min_neighbors
        #calculate the average f1-scores for the lower and greater epsilon values
        lower_average=lower_epsilon_f1scores/min_neighbors_iterations
        greater_average=greater_epsilon_f1scores/min_neighbors_iterations
        # if the f1score of the lower epsilon value is greater than set the  min_neighbors as the lower epsilon, otherwise set it as the greater epsilon
        if lower_average>greater_average:
            epsilon=lower_epsilon
        else:
            epsilon=greater_epsilon
    largest_f1score_position=0
    #go through each tuple collected to find the one with the greatest f1score
    for values_position in range(len(list_of_values)):
        if list_of_values[values_position][3]>list_of_values[largest_f1score_position][3]:
            largest_f1score_position=values_position
    #return the tuple with the largest f1score
    return list_of_values[largest_f1score_position]


def run(training, testing, actual_labels,min_neighbors, epsilon, verbose=False):
    best_values=tune_parameters(training, testing, actual_labels,min_neighbors, epsilon, verbose=False)
    return best_values[2]
