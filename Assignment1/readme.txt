CSEC 620 | Assignment 1 - TEAM 4
Nick Mangerian, Jacob Ruud, Hugo Tessier
-------------------------

1. DIRECTIONS
--------------
To run this code, The SMS Spam Collection v.1 dataset file (named SMSSpamCollection) should be placed in the same folder as the 4 python files main.py, functions.py, NaiveBayes.py, kNN.py.

To compare the performances of Naive Bayes and kNN classifiers, one should run the main.py Python script.

To obtain the figures presented in the report for kNN, one should open and run the file kNN.py Python file after having uncommented lines depending on the desired figures (after line 318): 
    - runSimpleKNN (line 321) to run a classification with kNN on the SMS dataset. 
        The distance metric and the number of neighbors (lines 210-211) can be modified.
        The feature_select_threshold (line 217) can be changed to see the impact of the feature_selection as described in the report.

    - tuneKNN (line 324) to see the plots that allow hyperparameters selection (choose number of neighbors and distance)
        The range of neighbors k_params_range (lines 229-230) can be modified narrow/wide.
        The range of distances can be modified (line 231)
        The score to plot can be modified (line 232)

    - comparethreshold (line 327) to see the scores graph for two different threshold (imbalance categories in variable to predict: see report 3.d)
        The thresholds thres1 and thres2 (lines 260-261) can be modified.
        The range of the number of neighbors, the distance and the score can be modified (lines 262-264) 


2. CODE STRUCTURE
---------------
The code functions are separated in four Python files which contain each a set of functions

- main.py
    main: function that compare k-NN and Mutlinomial Naive Bayes classifiers on the SMS Spam Collection dataset 

- functions.py
    - tokenize: function that tokenizes all raw messages of a text file
    - split_data: function that randomly split data into two complementary sets with sizes based on ratio
    - category_balance: function that prints the balance between categories in a dataset 

- NaiveBayes.py
    - preprocessor: preprocess of a dataset in TF-IDF in order to apply Naive Bayes classifier
    - classify: function that classifies a new SMS based on the probability dictionnary
    - run: function that runs Multinomial Naive Bayes classifier

- kNN.py
    - select_features: selection of the words (features) to keep
    - preprocess: preprocess of a dataset in TF-IDF in order to apply KNN classifier
    - run: function that runs KNN classifier (can handle multiple number of neighbors and distances)

    # Additional functions 
    - plot_accuracy: plot a score metric as a function of k and an additional variable
    - runSimpleKNN: Run a KNN on the SMS dataset
    - tuneKNN: function that allows hyperparameters tuning by plotting score graphs (choose number of neighbors and distance)
    - comparethreshold: plot scores graph for two different threshold (imbalance categories in variable to predict: see report)
