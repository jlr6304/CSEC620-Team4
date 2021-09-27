import numpy as np
import sys
import time
import sklearn



def find_points_in_epilsion(distances_of_points, epsilon, point_position):
    points_in_epilion = []
    for new_point_position in range(0, len(distances_of_points)):
        if (distances_of_points[point_position][new_point_position] < epsilon):
            points_in_epilion.append(new_point_position)
    return points_in_epilion

def check_if_in_cluster(clusters,point_position):
    in_cluster=False
    for points_of_cluster in clusters:
        if point_position in points_of_cluster:
            in_cluster = True
    return in_cluster

def create_cluster(distances_of_points, point_position,min_neighbors, epsilon, points_in_cluster):
    print("________________________________________________--")
    print("point_position: ",point_position)
    print("points_in_cluster: ", points_in_cluster)
    print("epsilon: ", epsilon)
    points_in_epilsion = find_points_in_epilsion(distances_of_points, epsilon,point_position,)
    print("points_in_epilsion: ", points_in_epilsion)
    prevoius_points=points_in_cluster+points_in_epilsion
    for point_position in points_in_epilsion:
        print("points_in_cluster: ", points_in_cluster)
        print("point_position: ", point_position)
        if point_position not in points_in_cluster:
            new_points_in_cluster=create_cluster(distances_of_points, point_position,min_neighbors, epsilon,prevoius_points)
            for new_point_position in new_points_in_cluster:
                if new_point_position not in points_in_cluster:
                    points_in_cluster.append(new_point_position)
    return points_in_cluster


def run(training, testing, min_neighbors, epsilon):
    #np.set_printoptions(threshold=sys.maxsize)
    #print("2nd collection of metrics:",training[1])
    #print( "3rd faeture(metric) of 2nd collectionfo metricos:", training[1][2])
    print("distances:")
    distances_of_points=sklearn.metrics.pairwise.euclidean_distances(training,training)
    print(distances_of_points)
    #print("distance from 2nd point to all other points:",distances_of_points[1])
    #print("distance from 2nd point to 3rd point:",distances_of_points[1][2])
    clusters=[]
    print("create clusters")
    clusters=[]
    for point_position in range(0,len(distances_of_points)):
        if check_if_in_cluster(clusters,point_position)==False:
            if len(find_points_in_epilsion(distances_of_points, epsilon, point_position)) >=min_neighbors:
                points_in_cluster=[]
                points_in_cluster.append(point_position)
                new_cluster=create_cluster(distances_of_points, point_position,min_neighbors, epsilon, points_in_cluster)
                clusters.append(new_cluster)
                print("new cluster:")
                print(new_cluster)
    print("clusters:")
    print(clusters)
    print("dinsacnatens done")

