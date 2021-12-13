from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# -- run SVM
def run_SVM(X_train, Y_train, X_test, Y_test, kernel='rbf', C=1, degree=3, gamma='scale'):
    classifier = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, target_names=["Benign", "Malicious"]))

    np.save("data/pred_labels_SVM.npy", Y_pred)
    return classifier.score(X_test, Y_test)

# -- tune SVM
def tune_SVM(X_train, Y_train):
    X_train, X_vali, Y_train, Y_vali = train_test_split(X_train, Y_train, test_size=.3)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    c_range = np.arange(2, 16, 4)/10
    degree_range = np.arange(1, 9)
    gamma_options = ['scale', 'auto']
    scores_linear = np.zeros(len(c_range))
    scores_poly = np.zeros((len(c_range), len(degree_range)))
    scores_rbf = np.zeros((len(c_range), len(gamma_options)))
    scores_sigmoid = np.zeros(len(c_range))
    for i, C in enumerate(c_range):
        score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel_options[0], C)
        scores_linear[i] = score
        for j, degree in enumerate(degree_range):
            score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel_options[1], C, degree)
            scores_poly[i, j] = score
        for k, gamma in enumerate(gamma_options):
            score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel_options[2], C, 3, gamma)
            scores_rbf[i, k] = score
        score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel_options[3], C)
        scores_sigmoid[i] = score

    # print("linear svm ideal hyperparameters", max(scores_linear, key=scores_linear.get), max(scores_linear.values()))
    # print("polynomial svm ideal hyperparameters", max(scores_poly, key=scores_poly.get), max(scores_poly.values()))
    # print("rbf svm ideal hyperparameters", max(scores_rbf, key=scores_rbf.get), max(scores_rbf.values()))
    # print("sigmoid svm ideal hyperparameters", max(scores_sigmoid, key=scores_sigmoid.get), max(scores_sigmoid.values()))

    # plot scores
    plt.plot(c_range, scores_linear)
    plt.title("Linear SVM hyperameter tuning")
    plt.xlabel("C-value")
    plt.ylabel("F1-score")
    plt.show()
    df = pd.DataFrame(scores_poly, index=c_range, columns=degree_range)
    sns.heatmap(df, annot=True, fmt='.3f')
    plt.title("F1-score")
    plt.xlabel("polynomial degree")
    plt.ylabel("C-value")
    plt.show()
    df = pd.DataFrame(scores_rbf, index=c_range, columns=gamma_options)
    sns.heatmap(df, annot=True, fmt='.3f')
    plt.title("F1-score")
    plt.xlabel("gamma option")
    plt.ylabel("C-value")
    plt.show()
    plt.plot(c_range, scores_sigmoid)
    plt.title("Sigmoid SVM hyperparameter tuning")
    plt.xlabel("C-value")
    plt.ylabel("F1-score")
    plt.show()


if __name__ == '__main__':
    # -- Load data
    X_train = np.load("data/train_set.npy", allow_pickle=True)
    Y_train = np.load("data/train_labels.npy", allow_pickle=True)

    X_test = np.load("data/test_set.npy", allow_pickle=True)
    Y_test = np.load("data/test_labels.npy", allow_pickle=True)

    # -- Run SVM
    print("F1-score: ", run_SVM(X_train, Y_train, X_test, Y_test))

    # -- Tune SVM
    # tune_SVM(X_train, Y_train)