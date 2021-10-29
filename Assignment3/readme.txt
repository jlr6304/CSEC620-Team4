CSEC 620 | Assignment 3 - TEAM 4
Nick Mangerian, Jacob Ruud, Hugo Tessier
-------------------------

1. DIRECTIONS
--------------
To run the code, the features_vectors files file have to be put in a ./data/feature_vectors/ folder and the sha256_family.csv file have to be put in a ./data/ folder. One could also unzip the preprocessed_data.zip file to have the family.npy, features.npy and samples.npy in the ./data/ folder. The current folder for execution should contain the Python scripts: preprocessing.py, functions.py, SVM.py, SVMsklearn.py, and multiclass_SVM.py.

The first step will consist in preprocessing the data by running the preprocessing.py file. It stores an array of the features (samples.npy), the features names (features.npy) and the family of the samples in samples.npy (family.npy) which corresponds to Benign or the family of the malware.

Once the preprocessing is done (one could also unzip the files from preprocessing_data.zip, SVM binary classification is done in the SVM.py script:
- run_SVM (line 253): to run the SVM classifier and obtain the performances and loss evolution with the number of epochs
- run_hyperparameter_tuning (line 259): to obtain the plots related to hyperparameters tuning, one could comment/uncomment lines 214 and 215 to change between narrow and wide range.
- run_feature_importance (line 256): to obtain the results for the importance of the features.

To obtain the results of the sklearn SVM one should run the SVMsklearn.py script.

To obtain the results of the multiclass SVM one should run the multiclass_SVM.py script.

2. CODE STRUCTURE
---------------
The code functions are separated into five Python files providing a set of functions described below. Further explanation on the functions such as parameters and functioning are given as comments in the source code.

- preprocessing.py
get_samples: Function that preprocesses the raw data into a dataset of the features of malwares & benign samples

- functions.py
	to_binary_categories: converts nominal categories into -1, 1 categories based on its classify_func parameter
	Recreate_categories: recreates the nominal categories from a {-1, 1}-array categories based on its recreate_func parameter
	split_train_test: splits the dataset into training/testing samples based on the given ratios
	import_data: imports and returns the preprocessed data
	confusion_matrix: prints the confusion matrix from the given labels (can be multicategorical)
	f1_score: computes the F1-score from the given labels (printing the result is optional)
	accuracy: computes the accuracy of the given labels (printing the result is optional)

- SVM.py
	fit: fits an hyperplane to the training set 
	predict: predicts the labels of the set samples based on the SVM hyperplane 
	feature_importance: function that prints the `n` most and least important features based on their weights and description on them
	run_SVM: runs the SVM binary classifier on the data
	run_feature_importance:  script related to the features importance
	run_hyperparameter_tuning: script related to the hyperparameters tuning

- SVMsklearn.py
	linearSVM: processes a linear SVM classifier using the `sklearn` package
	radialSVM: processes a radial SVM classifier using the `sklearn` package

- multiclass_SVM.py
	split_train_test_malware: splits the `df` dataset into training/testing samples based on the 70/30 train/test split.
	one_vs_all: processes the one-vs-all classifier for multi categorical data.
	one_vs_one: processes the one-vs-one classifier for multi categorical data.

3. RESOURCES
---------------
For this assignment, we used external library that are listed below:
numpy for array programing: https://numpy.org/doc/
pandas for dataframe usage: https://pandas.pydata.org/
torch for tensor usage and gradient descent https://pytorch.org/
sklearn for comparison for its implemented Support Vector Classifier: https://scikit-learn.org/stable/
matplotlib for plotting graphs: https://matplotlib.org/


