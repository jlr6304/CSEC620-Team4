import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run_LR(X_train, y_train, X_test, y_test, n_trees=5, feature_subcount=3, max_depth=12, min_node=1):
    """
    ### Run Logistic Regression classifier on the dataset and save the predicted labels as a file

    """
    # ---- Logistic Regression Initialization
    model= LogisticRegression()

    # ---- Training
    model.fit(X_train, y_train)

    # ---- Predict on testing
    y_pred = (model.predict(X_test))

    np.save("data/pred_labels_LR.npy", y_pred)
    _f1_score = f1_score(y_test, y_pred)
    print("f1_score =", _f1_score)
    return y_pred


if __name__ == '__main__':
    # -- Load data
    X_train = np.load("data/train_set.npy", allow_pickle=True)
    y_train = np.load("data/train_labels.npy", allow_pickle=True)

    X_test = np.load("data/test_set.npy", allow_pickle=True)
    y_test = np.load("data/test_labels.npy", allow_pickle=True)

    # -- Run Logistic Regression
    run_LR(X_train, y_train, X_test, y_test, n_trees=5, feature_subcount=3, max_depth=12, min_node=1)
