CSEC 620 | Assignment 4 - TEAM 4
Nick Mangerian, Jacob Ruud, Hugo Tessier
-------------------------

1. DIRECTIONS
--------------
To run the code the data files have to be stored in the ./iot_data folder. For the execution, the current folder has to contain the three Python scripts: classify.py, DecisionTree.py and, RandomForest.py.

All the elements presented in the reports have been created using functions of the classify.py script. This script has to be run with the following command: python classify.py -r ./iot_data
To obtain the results related to the Random Forest classification, one has to uncomment lines 474.
To obtain the results related to the comparison between Decision Tree and Random Forest classifiers, one has to uncomment lines 478.
To obtain the elements used to compute the node importance, one has to uncomment lines 482.
To obtain the graphs related to the Random Forest hyperparameters tuning, one has to uncomment lines 486. The hyperparameter tuned can be changed between lines 397 and 401. The range in line 423 has to correspond to the hyperparameter range chosen.

To test the execution of the DecisionTree and RandomForest classifiers one can run the RandomForest.py script (its results aren’t relevant as it tests on randomly generated samples).

The images used for the reports (including the confusion matrix) can be found in the ./img folder.


2. CODE STRUCTURE
---------------
The code functions are separated into three Python files providing a set of functions described below. Further explanation on the functions such as parameters and functioning are given as comments in the source code.

- classify.py
	parse_args: parse arguments
	load_data: load json feature files produced from feature extraction
	do_stage_0: process each multinomial feature using naive bayes
	randomforest_classification: perform the random forest classification procedure
	algorithm_comparison: compare the performances of Random Forest and Decision Tree classifiers
	node_importance: display the labels and the impurity of a node and its children in order to compute the node importance
	hyperparameter_tuning: method to tune the Random Forest classifier hyperparameters
	main: perform main logic of the program


- DecisionTree.py
	Node:__init__: builder of the Node object
	Node:split: method to split the node (splits the node if possible and create and split children nodes (recursive function))
	Node:isLeaf: test if the node is a leaf
	Node:predict: predict the label of a sample if it is a leaf, otherwise predict the sample on the left or right child node depending on the feature and split threshold (recursive function)
	gini_impurity: compute the Gini impurity of a node based on its labels
	best_split_impurity: find the split threshold to minimize the Gini impurity if the node is splitted on the given feature


- RandomForest.py:
	train: train a Random Forest model based on the data
	predict: predict the labels of a set based on a Random Forest model


3. RESOURCES
---------------

The classify.py script used have been created during the following study
A. Sivanathan et al., "Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics," in IEEE Transactions on Mobile Computing, vol. 18, no. 8, pp. 1745-1759, 1 Aug. 2019, doi: 10.1109/TMC.2018.2866249.

For this assignment, we used external library that are listed below:
numpy for array programing: https://numpy.org/doc/
pandas for dataframe usage: https://pandas.pydata.org/
sklearn for comparison for its scores functions: https://scikit-learn.org/stable/
matplotlib for graphs: https://matplotlib.org/
