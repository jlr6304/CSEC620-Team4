CSEC 620 | Project - TEAM 4
Nick Mangerian, Jacob Ruud, Hugo Tessier
-------------------------

1. DIRECTIONS
--------------

The raw data for this study can be found at the following link http://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html

To run the code the data files have to be stored in the ./data folder. All the other Python scripts and notebooks must be in the root directory. These files are: feature_selection.ipynb, preprocessing.ipynb, LogisticRegression.py, RandomForest.py, SVM.py, NeuralNetwork.py, model_selection.ipynb.
Note: The notebooks can also be opened in their html format which are easier to open.

If working on the raw data. The metadata pdf extraction is made with exiftool with the following command line: exiftool.exe -r -csv ./data/benign ./data/malware > metadata.csv. It creates a new file metadata.csv. 

Then, to process the feature selection and follow it step by step, one should execute the cells of the Python notebook feature_selection.ipynb. It creates a cleaned version of metadata.csv called new_metadata.csv. This script also contains all the cleaning steps explained in the report.

The preprocessing of this new table is done in the notebook preprocessing.ipynb. It splits the data into a train and a test set and studies the normalization. It creates 5 new files: train_set.npy, train_labels.npy, test_set.npy, test_labels.npy, and columns.npy. Those are numpy arrays, it’s the reason why the columns are stored separately.

From these files, one can run all the classifier scripts. 
	LogisticRegression.py: runs a logistic regression classifier, no tuning is required for this model. The predicted labels for the test set are stored in data/pred_labels_LR.npy.
	
RandomForest.py: runs a RandomForest classifier. The tuning graphs are created by uncommenting the line 74 of the script. One can choose the hyperparameter to tune by modifying the lines 38 to 41. The labels for the best Random Forest classifier are stored in data/pred_labels_RF.npy.
	
SVM.py:  runs a SVM classifier and Kernel SVM classifiers. The tuning graphs are created by uncommenting the line 95. The labels for the best Linear SVM and Kernel SVMs classifiers are stored in data/pred_labels_SVM.npy.


NeuralNetwork.py: runs a 1 hidden layer Neural Network classifier. The tuning graphs are created by uncommenting the line 131 of the script. One can also check if the weights seem to converge by uncommenting the line 134. The labels of the test set for the best Neural Network are stored in data/pred_labels_NN.npy.

The model selection of the trained classifiers is done in the notebook model_selection.ipynb. It contains the code to represent the ROC curves, performance matrix, and confusion matrix presented in the report. It also contains the code related to the Logistic Regression feature importance analysis.

2. CODE STRUCTURE
---------------

- feature_selection.ipynb: a notebook that explains step by step how the features are selected for the final dataset and the data cleaning and data binning process.

- preprocessing.ipynb: a notebook that explains step by step how the dataset is processed in order to be given to sklearn libraries (it includes normalization and one-hot encoding).

- LogisticRegression.py
	run_LR: run Logistic Regression classifier on the dataset and save the predicted labels as a file.

- RandomForest.py
	run_RF: run Random Forest classifier on the dataset and save the predicted labels as a file.
	tune_RF: tune Random Forest hyperparameters of the tree.

- SVM.py
	run_SVM: train Linear SVM and Kernel SVM classifier on the train set and test it on the test set.
	tune_SVM: tune SVM and Kernel SVM classifiers. 

- NeuralNetwork.py
	f1_m: computes F1-score using Keras syntax.
	run_NN: run 1 hidden layer Neural Network on the dataset and save the predicted labels as a file.
	tune_NN: tune Neural Network activation function and number of nodes.
	check_convergence: displays the convergence of the F1-score with the epochs and depending on batch size.

- model_selection.ipynb: a notebook that justifies the model selection based on the classifiers performances.


3. RESOURCES
---------------
- exiftool for metadata extraction: https://github.com/exiftool/exiftool
- pandas for the usage of dataframes: https://pandas.pydata.org/
- sklearn for their ML algorithm implementation and the model comparison with its scores functions: https://scikit-learn.org/stable/
- matplotlib and seaborn for showing graphs: https://matplotlib.org/ & https://seaborn.pydata.org/

