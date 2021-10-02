import numpy as np
import sys
import time
import sklearn


def find_points_in_epsilon(distances_of_points, epsilon, point_position):
    points_in_epilion = []
    for new_point_position in range(0, len(distances_of_points)):
        if (distances_of_points[point_position][new_point_position] < epsilon):
            points_in_epilion.append(new_point_position)
    return points_in_epilion


def determine_core_points(distances_of_points,epsilon,min_neighbors, verbose=False):
    print("\tdetermining core points in training set")
    
    core_points_positions=[]
    for point_position in range(0,len(distances_of_points)):
        if len(find_points_in_epsilon(distances_of_points, epsilon,point_position))>=min_neighbors:
            if verbose:
                print("core point found")
            core_points_positions.append(point_position)
    return core_points_positions


def test(testing, training, core_points, epsilon, verbose = False):
    print("\tanomaly detection on testing set")
    
    distances_from_core_points=sklearn.metrics.pairwise.euclidean_distances(testing, core_points)
    # print(distances_from_core_points) # print the distance with core points

    labels = []
    for sample in distances_from_core_points:
        anomaly=True
        for distance in sample:
            if distance <=epsilon:
                anomaly=False
                break
        labels.append(1 if anomaly else 0)
        
        # Print the category
        if verbose:
            if anomaly==True:
                print("anomaly")
            if anomaly==False:
                print("not anomaly")

    return labels


def run(training, testing, min_neighbors, epsilon, verbose = False):

    distances_of_points=sklearn.metrics.pairwise.euclidean_distances(training,training)
    # print(distances_of_points) # print the distance between training points
    
    core_point_positions=determine_core_points(distances_of_points,epsilon,min_neighbors, verbose)
    core_points=[]
    
    for position in core_point_positions:
        core_points.append(training[position])
    
    return test(testing, training, core_points, epsilon, verbose)
