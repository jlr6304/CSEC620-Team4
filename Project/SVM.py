from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

# -- run SVM
def run_SVM(X_train, Y_train, X_test, Y_test, kernel='rbf', C=1, degree=3, gamma='scale'):
    classifier = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))

    np.save("data/pred_labels_SVM.npy", Y_pred)
    return classifier.score(X_test, Y_test)

# -- tune SVM
def tune_SVM(X_train, Y_train):
    X_train, X_vali, Y_train, Y_vali = train_test_split(X_train, Y_train, test_size=.3)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    c_range = np.arange(2, 16, 4)/10
    scores_linear = {}
    scores_poly = {}
    scores_rbf = {}
    scores_sigmoid = {}
    for kernel in kernel_options:
        for C in c_range:
            if kernel == 'linear':
                score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel, C)
                scores_linear[C] = score
            elif kernel == 'poly':
                degrees = np.arange(1, 9)
                for degree in degrees:
                    score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel, C, degree)
                    scores_poly[(C, degree)] = score
            elif kernel == 'rbf':
                gamma_options = ['scale', 'auto']
                for gamma in gamma_options:
                    score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel, C, 3, gamma)
                    scores_rbf[(C, gamma)] = score
            elif kernel == 'sigmoid':
                score = run_SVM(X_train, Y_train, X_vali, Y_vali, kernel, C)
                scores_sigmoid[C] = score
    print("linear svm ideal hyperparameters", max(scores_linear, key=scores_linear.get), max(scores_linear.values()))
    print("polynomial svm ideal hyperparameters", max(scores_poly, key=scores_poly.get), max(scores_poly.values()))
    print("rbf svm ideal hyperparameters", max(scores_rbf, key=scores_rbf.get), max(scores_rbf.values()))
    print("sigmoid svm ideal hyperparameters", max(scores_sigmoid, key=scores_sigmoid.get), max(scores_sigmoid.values()))


if __name__ == '__main__':
    # -- Load data
    X_train = np.load("data/train_set.npy", allow_pickle=True)
    Y_train = np.load("data/train_labels.npy", allow_pickle=True)

    X_test = np.load("data/test_set.npy", allow_pickle=True)
    Y_test = np.load("data/test_labels.npy", allow_pickle=True)

    # -- Run SVM
    run_SVM(X_train, Y_train, X_test, Y_test)

    # -- Tune Neural Network
    # tune_SVM(X_train, Y_train)

    # -- Neural Network convergence
    # check_convergence(X_train, y_train, n_epochs=200, batch_size=50)