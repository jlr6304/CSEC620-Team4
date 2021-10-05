CSEC 620 | Assignment 2 - TEAM 4
Nick Mangerian, Jacob Ruud, Hugo Tessier
-------------------------

1. DIRECTIONS
--------------
To run the code, The KDD’99 NIDS datasets file (named testing_attack.npy, testing_normal.npy, and training_attack.npy) should be placed in a subfolder ./data/ of the folder that contains the 5 python files Assignment2.py, PCA.py, functions.py, DBSCAN.py, and Kmeans.py.

To compare the performances of DBSCAN and k-Means anomaly detection algorithms, one should run the Assignment2.py script.

To obtain the first figures presented in the report (for PCA), one should open and run thePCA.py  file after having uncommented lines depending on the desired figures (after line 142):
    - compare2D (line 151): to simultaneously plot normal behavior and attacks in a 2D space
    - compare3D (line 152): to simultaneously plot normal behavior and attacks in a 3D space 
    - tune_number_of_components (line 155): to plot the explained variance ratio as a function of the number of components in PCA
    - reduce_dimensions (line 158): reduce the number of features of the datasets, the number of features to keep can be changed

To obtain the plot which concerns epsilon starting value and understand our data has been imported, one should open and run the functions.py file.
    - get_data (line 147): one can change the number of samples in the testing dataset and the ratio between attacks and normal behavior. 
    - score (line 150): gives an example of how the performance metrics are calculated one can change the score function
    - epsilon_start(line 153): allows to plot the “Choosing epsilon starting value” graph of the report

The performances obtained for the DBSCAN hyperparameter tuning can be found by opening and running the DBSCAN.py file.

The performances obtained for the k-Means hyperparameter tuning can be found by opening and running the Kmeans.py file.


2. CODE STRUCTURE
---------------
The code functions are separated into five Python files providing a set of functions described below. Further explanation on the functions such as parameters are given as comments in the source code.

- Assignment2.py
    main: function that compares k-Means and DBSCAN performances on the KDD’99 NIDS dataset

- functions.py
    - get_data: import data to cluster from the data repository
    - score: compute and print score metrics between predicted and actual labels
    - epsilon_start: plots the sorted distances of the nearest neighbor for all the samples of the PCA projected training set

- PCA.py
    - tune_number_of_components: plot the explained variance ratio as a function of the number of components
    - reduce_dimensions: reduce the number of features by applying PCA
    - compare2D: comparison of the testing normal and testing attack datasets in the 2 first dimensions of the PCA feature space
    - compare3D: comparison of the testing normal and testing attack datasets in the 3 first dimensions of the PCA feature space

- DBSCAN.py
    - find_points_in_epsilon: finds the samples that are inside a circle of radius epsilon
    - determine_core_points: determines the DBSCAN core points from the training dataset
    - test: classifies the testing dataset samples as normal behavior or attacks
    - get_predicted_labels: computes the distance with the core points dans classifies the testing samples using the `test` function
    - run: runs the DBSCAN anomaly detector
    # additional functions
    - find_f1score: computes F1-score based on labels
    - generate_scores: prints and returns the score of an hyperparameters combination
    - tune_parameters: tunes the epsilon and minimal number of neighbors hyperparameters

- kMeans.py
    - create_clusters: create clusters and find centroids of normalized data
    - run: runs the k-means anomaly detector
    # additional function
    - tune_hyperparameters: function to test out multiple values for t and k and determine the combination of these values which yields the highest f1-score


3. RESOURCES
---------------
For this assignment, we used 
numpy for array programing: https://numpy.org/doc/
sklearn for distance computation: https://scikit-learn.org/stable/
matplotlib for plotting graphs: https://matplotlib.org/
