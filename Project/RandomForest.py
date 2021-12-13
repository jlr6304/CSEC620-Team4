import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def run_RF(X_train, y_train, X_test, y_test, n_trees=150, feature_subcount=3, max_depth=12, min_node=1):
    """
    ### Run Random Forest classifier on the dataset and save the predicted labels as a file

    """
    # ---- Random Forest Initialization
    model= RandomForestClassifier(n_estimators=n_trees, max_features=feature_subcount, max_depth=max_depth,min_samples_leaf=min_node)

    # ---- Training
    model.fit(X_train, y_train)

    # ---- Predict on testing
    y_pred = (model.predict(X_test))

    np.save("data/pred_labels_RF.npy", y_pred)
    _f1_score = f1_score(y_test, y_pred)
    print("f1_score =",_f1_score)
    return y_pred


def tune_RF(X_train,y_train):
    """
    ### Tune Random Forest hyperparameters of the tree

    The tuning is done on a validation set different from the testing set
    """

    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=.3)

    # Hyperparameters list (single values are best hyperparameters)
    n_trees_range =  np.arange(10, 500, 20) # [150]
    feature_subcount_range = [3]  # np.arange(1, 10)
    max_depth_range = [12] # np.arange(1, 20)
    min_node_range = [1]  # np.arange(1, 10)

    score = []
    for n_trees in n_trees_range:
        for feature_subcount in feature_subcount_range:
            for max_depth in max_depth_range:
                for min_node in min_node_range:
                    clf = RandomForestClassifier(n_estimators=n_trees, max_features=feature_subcount, max_depth=max_depth, min_samples_leaf=min_node).fit(X_train, y_train)
                    y_pred = clf.predict(X_vali)
                    _f1_score = f1_score(y_vali, y_pred)

                    score.append(_f1_score)  # add f1-score to the score array

    # Graph the results
    range = np.array(n_trees_range)
    score = np.array(score)

    plt.plot(range, score)
    plt.show()


if __name__ == '__main__':
    # -- Load data
    X_train = np.load("data/train_set.npy", allow_pickle=True)
    y_train = np.load("data/train_labels.npy", allow_pickle=True)

    X_test = np.load("data/test_set.npy", allow_pickle=True)
    y_test = np.load("data/test_labels.npy", allow_pickle=True)

    # -- Run Random Forest
    run_RF(X_train, y_train, X_test, y_test, n_trees=200, feature_subcount=3, max_depth=12, min_node=1)
    
    # -- Tune Random Forest
    # tune_RF(X_train, y_train)

    
