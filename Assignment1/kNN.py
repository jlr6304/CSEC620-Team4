# Modules import
import numpy as np # to fasten utilization and computation with arrays 
import matplotlib.pyplot as plt # to create plots 

from functions import tokenize, split_data, score, category_balance

"""
Selection of the words (features) to keep
    parameters
        tokens: unpreprocessed (tokenized) dataset of SMS
        threshold: minimum number of occurence of a word (feature selection)
    return
        feature_vector: words to consider
        IDF: IDF for the words in feature_vector
"""
def select_features(tokens, threshold = 2):
    tokens = [tokens[i][1:] for i in range(len(tokens))]  # drop the SMS category

    feature_vector = list(set().union(*(message for message in tokens))) # list of unique words in all SMSs
    feature_vector = np.array(feature_vector)
    n_features = len(feature_vector)

    index = dict(zip(feature_vector, range(n_features))) # keep memory of the indexes of the words
    
    # Count number of occurence for each word
    counts = np.zeros(n_features, dtype=float) # number of SMS which contains at least one occurence for each word
    for message in tokens:
        for w in set(message): # using set remove duplicates
            counts[index[w]] +=1

    # Feature selection (remove rare words based on threshold)
    keep = [k for k in range(n_features) if counts[k] >= threshold] # indices of words to keep
    print('number of features = ', len(keep))

    # Updated feature vector & counts with kept words
    feature_vector = feature_vector[keep]
    counts = counts[keep]

    # Compute IDF
    N_SMS = len(tokens)
    IDF = np.log(N_SMS) - np.log(counts)

    return feature_vector, IDF

"""
Preprocess of a dataset in TF-IDF in order to apply KNN classifier
    parameters
        tokens: unpreprocessed (tokenized) dataset of SMS
        feature_vector: unique words to consider 
        IDF: IDF for each word in the feature vector
    return
        dataset: preprocessed dataset of SMS
        labels: labels of the 
"""
def preprocess(tokens, feature_vector, IDF):
    
    labels = [tokens[i][0] for i in range(len(tokens))]  # return labels of all SMS and 
    tokens = [tokens[i][1:] for i in range(len(tokens))] # remove labels from tokens

    N_SMS = len(tokens) # number of SMS
    n_features = len(feature_vector) # number of features
    index = dict(zip(feature_vector, range(n_features))) # keep memory of the indexes of the words
    
    # Convert each message in its TF-IDF form
    dataset = np.zeros((N_SMS, n_features), dtype=float) # create and fill the dataset with zeros
    
    for k,message in enumerate(tokens):
        # Count word frequency per SMS 
        words, counts = np.unique(message, return_counts=True) # return unique words and their count
        counts = dict(zip(words, counts))

        # Add information to dataset
        for w in words:
            if w in feature_vector:
                dataset[k, index[w]] = counts[w]

        # Compute TF per SMS
        n = len(message)
        TF = dataset[k,]/n
        
        # Compute TF-IDF per SMS
        dataset[k,] = TF * IDF
        
    return dataset, labels


"""
Function that runs KNN classifier (can handle multiple number of neighbors and distances)
    parameters
        training_set: [list]  unpreprocessed (tokenized) train dataset of SMS
        test_set: [list]  unpreprocessed (tokenized) test dataset of SMS
        k_param: number of neighbors (can be an iterable)
        distances: distance metric (can be an iterable) values must be in "euclidean", "manhattan", "inf_norm" 
        feature_select_threshold: minimum number of apparition of a word
        balance_threshold: can be ajusted in case of imbalanced category in the variable to predict 
    return
        pred_labels: [list] true labels for the test set
        true_labels: [list] predicted labels for the test set
"""
def run(training_set, test_set, k_param=3, distances="euclidean", feature_select_threshold=2, balance_threshold=.5):

    # Feature selection 
    feature_vector, IDF = select_features(training_set, threshold = feature_select_threshold)

    # Preprocessing
    training_set, training_labels = preprocess(training_set, feature_vector, IDF)
    test_set, true_labels = preprocess(test_set, feature_vector, IDF)

    # Pretreat the input parameters
    # - number of neighbors: creates an array of the k to test
    if type(k_param) == int: 
        k_param = [k_param]
    
    # - distances: creates an array of the distances to test
    if type(distances) == str:
        distances = [distances]

    distance_function = []
    for d in distances:
        # euclidean distance function
        if d == "euclidean":
            sum_by_row  = lambda df: np.apply_along_axis(lambda x: x[x>0].sum(), 1, df) # we only sum positive values to fasten the computation
            distance_function.append( lambda x,y: sum_by_row( (x-y)**2) )
        # manhattan distance function 
        elif d == "manhattan":
            sum_by_row  = lambda df: np.apply_along_axis(lambda x: x[x>0].sum(), 1, df) # we only sum positive values to fasten the computation
            distance_function.append( lambda x,y: sum_by_row( np.abs(x-y)) )
        # infinity norm distance function
        elif d == "inf_norm":
            max_by_row = lambda df: np.apply_along_axis(lambda z: max(z), 1, df)
            distance_function.append( lambda x,y:  max_by_row(np.abs(x-y)) )

    # Convert labels to binary
    to_binary = lambda x: 0 if x == "ham" else 1
    training_labels = list(map(to_binary, training_labels))
    
    print("training dataset shape =", training_set.shape)
    print("test dataset shape =", test_set.shape)

    # Prediction of each message type
    pred_labels = np.full((len(test_set), len(distance_function), len(k_param)), fill_value="", dtype=object)
    
    print("Number of classified SMS:")
    for i,message in enumerate(test_set):

        if i%50 == 49:
            print("\t", i+1, "rows")
        
        # compute distances
        for j, dist_func in enumerate(distance_function): 
            
            distances = dist_func(training_set, message)
            
            # Predict label
            for k,k_val in enumerate(k_param):
                k_min_distances_indices = np.argpartition(distances, k_val)[:k_val]

                mean = np.array(training_labels)[k_min_distances_indices].mean()
                prediction = "spam" if mean>=balance_threshold else "ham"
            
                pred_labels[i, j, k] = prediction

    if len(k_param)==1 and len(distance_function)==1:
        pred_labels = np.reshape(pred_labels, (len(pred_labels),))
    
    return pred_labels, true_labels




# ---------------------------------------------------------------------------------
# ----------------------FUNCTIONS TO REPRODUCE REPORT RESULTS----------------------
# ---------------------------------------------------------------------------------



"""
Plot a score metric as a function of k and an additional variable
    parameters
        scores: score for each hyperparameter (2nd dimension corresponds to labels parameter)
        k_params_range: [list] number of neighbors
        labels: [list] additional labels
        score_label: name of the second variable
"""
def plot_score(scores, k_params_range, labels, score_label):
    ax = plt.subplot(111)
    for j in range(scores.shape[0]):
        ax.plot(k_params_range, [i*100 for i in scores[j,:]], '--+', label = labels[j])
    
    plt.legend()
    plt.title(score_label + " as a function of $k$")
    plt.xlabel('$k$ number of neighbors'); plt.ylabel(score_label + ' (%)')
    plt.show()


"""
Run a KNN on the SMS dataset
"""
def runSimpleKNN():
    tokens = tokenize("SMSSpamCollection")
    # ------------------ Separate training, validation and test set
    training_validation_set, test_set = split_data(tokens, ratio = .6)
    training_set, validation_set = split_data(training_validation_set, ratio = .8)

    print("----------------------------------------")
    print("training dataset size:\t", len(training_set))
    print("validation dataset size:", len(validation_set))

    # ------------------ run kNN
    k_optim = 4
    distance_optim = "euclidean"
    predicted_labels , true_labels = run(
        training_set, 
        training_set,
        k_param = k_optim,
        distances = distance_optim,
        feature_select_threshold = 2
    )

    # ------------------ print score
    score(predicted_labels, true_labels, ["confusion_mat","accuracy"], return_value=False)


"""
Function that allows hyperparameters tuning by plotting score graphs (choose number of neighbors and distance) 
"""
def tuneKNN():
    # ------------------ Tuning parameters
    # k_params_range = list(range(5, 205, 10))   # wide range
    k_params_range = list(range(2, 12, 1))       # narrow range
    distances_range = ["euclidean", "manhattan"] # "inf_norm" can be added
    score_name = "accuracy"

    # ------------------ Separate training, validation and test set
    tokens = tokenize("SMSSpamCollection")
    training_validation_set, test_set = split_data(tokens, ratio = .7)
    training_set, validation_set = split_data(training_validation_set, ratio = .8)

    predicted_labels , true_labels = run(
        training_set, 
        validation_set,
        k_param = k_params_range,
        distances = distances_range,
        feature_select_threshold = 2)

    # Compute accuracy score
    scores = np.zeros((predicted_labels.shape[1], predicted_labels.shape[2]))
    for j in range(scores.shape[0]):
        for k in range(scores.shape[1]):
            p = predicted_labels[:, j, k]
            scores[j,k] = score(p, true_labels, score_name, return_value=True)

    # Plot the accuracy score
    plot_score(scores, k_params_range, distances_range, score_label=score_name.capitalize())   

"""
Plot scores graph for two different threshold (imbalance categories in variable to predict: see report) 
"""
def comparethreshold():
    thres1 = .5
    thres2 = .3
    k_params_range = list(range(2, 12, 1)) # narrow range
    distance_optim = "euclidean"
    score_metric = "F1score"
    
    # ------------------ Separate training, validation and test set
    tokens = tokenize("SMSSpamCollection")
    training_validation_set, test_set = split_data(tokens, ratio = .6)
    training_set, validation_set = split_data(training_validation_set, ratio = .8)

    # ------------------ First threshold
    print("Classification with the first threshold")
    predicted_labels_1 , true_labels = run(
        training_set, 
        validation_set,
        k_param = k_params_range,
        distances = distance_optim,
        feature_select_threshold = 2,
        balance_threshold = thres1
    )

    predicted_labels_1 = predicted_labels_1[:,0,:] #drop the second axis used for distances comparison
    
    # ------------------ Second threshold
    print("Classification with the second threshold")
    predicted_labels_2 , true_labels = run(
        training_set, 
        validation_set,
        k_param = k_params_range,
        distances = distance_optim,
        feature_select_threshold = 2,
        balance_threshold = thres2
    )

    predicted_labels_2 = predicted_labels_2[:,0,:] #drop the second axis used for distances comparison

    # Compute accuracy score
    scores = np.zeros((len(k_params_range), 2))
    for k in range(scores.shape[0]):
        scores[k,0] = score(predicted_labels_1[:, k] , true_labels, score_metric, return_value=True)
        scores[k,1] = score(predicted_labels_2[:, k] , true_labels, score_metric, return_value=True)

    # Plot the score as a function of k for several thresholds
    ax = plt.subplot(111)
    ax.plot(k_params_range, [i*100 for i in scores[:,0]], '--+', label = f"thres1={thres1}")
    ax.plot(k_params_range, [i*100 for i in scores[:,1]], '--+', label = f"thres2={thres2}")

    plt.legend()
    plt.title(score_metric.capitalize() + " as a function of $k$ for different thresholds")
    plt.xlabel("$k$ number of neighbors"); plt.ylabel(score_metric)
    plt.show()


# -------------------------------------------------------------------------------------------------------
# MAIN FUNCTION TO TO REPRODUCE REPORT RESULTS
# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # -------------- Runs a KNN on the SMS dataset
    # runSimpleKNN()

    # -------------- Plots graphs to tune kNN (choose number of neighbors and distance) 
    # tuneKNN()

    # -------------- Compares threshold for imbalanced categories in dataset
    # comparethreshold()
    
    pass