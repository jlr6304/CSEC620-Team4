import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from functions import accuracy, distance_point_hyperplane, import_data, split_train_test, to_binary_categories, recreate_categories, confusion_matrix, f1_score, bold


def fit(train_set, train_labels, C=0.4, learning_rate=1e-2, n_epoch=100, eps = 1e-2):
    """
    #### Fits an hyperplane to the training set 
    parameters: 
        - train_set: set of samples to separate with an hyperplane
        - train_labels: labels of the previous samples in {-1, 1}
        - C: regularizing coefficient
        - n_epoch: maximum number of epochs of the Stochastic Gradient Descent
        - learning_rate: learning rate of the Stochastic Gradient Descent
        - eps: convergence threshold for the Stochastic Gradient Descent
    return:
        - hyperplane: the hyperplane parameters of the minimized model
        - loss_evol: the evolution of the loss
    """

    print(bold("Fitting SVM"))

    # tensor initialization
    x = torch.from_numpy(train_set).type(torch.float)
    t = torch.from_numpy(train_labels).type(torch.float)

    ones = torch.ones(t.size())
    zeros = torch.zeros(t.size())

    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    w = torch.randn(train_set.shape[1], requires_grad=True, dtype=torch.float)

    loss_evol = [] # keeps track of the evolution of the loss through the iterations

    # --- Stochastic Gradient Descent
    # Loop on epochs
    for e in range(n_epoch):
        print(f'Epoch {e+1}/{n_epoch}')
        
        # Loop on samples
        for i in range(train_set.shape[0]):
            # Hinge Loss
            y = torch.dot(x[i, :], w) - b
            HL = torch.max(torch.zeros(1), torch.ones(1) - t[i]*y)            
            # Margin loss
            ML = C*torch.norm(w)/2 # Frobenius norm is equivalent to euclidian norm for an array
                        
            # Loss function and backward
            loss = HL + ML
            loss.backward()

            # Update slope and intersect + reinitialize gradient
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad.zero_()
            b.grad.zero_()

        # Compute the loss after each epoch to store its evolution
        y = torch.matmul(x, w) - b.expand_as(t)
        HL = torch.maximum(ones - torch.mul(t,y), zeros).sum() # element-wise maximum
        ML = C*torch.norm(w)/2
        loss_evol.append((HL + ML).item())
        
        # Stop criterion based on the evolution of the loss
        if e>0:
            if np.abs(loss_evol[e]-loss_evol[e-1])/loss_evol[e-1]< eps:
                print("Algorithm converged")
                break

    # Convert tensors to arrays and float
    w = w.detach()
    b = b.detach().float()

    return {'slope':w, 'intercept':b}, loss_evol


def predict(set, hyperplane):
    """
    #### Predict the labels of the set samples based on the SVM hyperplane 
    parameters:
        - test_set: set of samples to categorize
        - hyperplane: SVM hyperplane to perform the classification
    """
    n_samples = set.shape[0]
    
    # Hyperplane information
    w = hyperplane['slope'] # slope
    b = hyperplane['intercept'].expand(n_samples) # intercept

    # Convert the dataset into a tensor
    set = torch.from_numpy(set).type(torch.float) 
    
    # Predict new labels
    labels = (torch.matmul(set, w) - b).sign()

    # Convert back to a np.array 
    labels = labels.detach().numpy().astype(int)
    return labels


def feature_importance(train, features, weights, n=10):
    """
    #### Function that prints the `n` most and least important features based on their weights and description on them 
    """
    weights = np.abs(weights.numpy()) # absolute value of the weights

    # Index of the most and least important features
    index_max = (-weights).argsort()[:n]
    index_min = (-weights).argsort()[-n:]

    # Extract data related to these features 
    train_max = pd.DataFrame(train[:,index_max], columns = features[index_max], dtype = object)
    train_min = pd.DataFrame(train[:,index_min], columns = features[index_min], dtype = object)

    # Print features importance with pandas.describe
    print(bold('Most important features:'))
    print(train_max.describe().transpose())

    print(bold('\nLeast important features:'))
    print(train_min.describe().transpose())


def run_SVM():
    """
    #### Run the SVM binary classifier on the data
    """
    # import data from the preprocessed dataset
    df, labels, features = import_data()
    
    # split into training and testing sets
    train_set, train_labels, test_set, test_labels , features, = split_train_test(
        df, labels, features, total_samples=4000, test_ratio=.3
    )

    # convert labels into -1 and 1
    train_labels = to_binary_categories(train_labels)
    test_labels = to_binary_categories(test_labels)

    # fit model on training set
    hyperplane, l = fit(
        train_set, train_labels, C=.2, learning_rate=1e-2, n_epoch=50, eps = 1e-2
    )

    # predict on testing set
    true_labels = test_labels
    pred_labels = predict(test_set, hyperplane)

    # Recreate user-friendly labels
    true_labels = recreate_categories(true_labels)
    pred_labels = recreate_categories(pred_labels)
    
    # Confusion matrix
    confusion_matrix(pred_labels, true_labels)

    # F1-score
    f1_score(pred_labels, true_labels, verbose=True)

    # Accuracy
    accuracy(pred_labels, true_labels, verbose=True)

    # Plot the evolution of the loss through the iterations
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.plot(np.arange(len(l)), np.array(l))
    plt.title("Loss as a function of the number of epochs")
    plt.xlabel("Number of epochs"); plt.ylabel("Loss function") 
    plt.show()

def run_feature_importance():
    """
    #### Script related to the features importance
    """
    df, labels, features = import_data()
    
    # split into training and testing sets
    train_set, train_labels, test_set, test_labels, features = split_train_test(
        df, labels, features, total_samples=4000, test_ratio=.3
    )

    # convert labels into -1 and 1
    train_labels = to_binary_categories(train_labels)
    test_labels = to_binary_categories(test_labels)

    # fit model on training set
    hyperplane, l = fit(
        train_set, train_labels, C=.2, learning_rate=1e-2, n_epoch=50, eps = 1e-2
        )

    # evaluate feature importance
    feature_importance(train_set, features, hyperplane['slope'])


def run_hyperparameter_tuning():
    """
    #### Script related to the hyperparameters tuning
    """
    # import data from the preprocessed dataset
    df, labels, features = import_data()
    
    # split into training and testing sets
    train_set, train_labels, test_set, true_labels , features, = split_train_test(
        df, labels, features, total_samples=4000, test_ratio=.3
    )

    # convert labels into -1 and 1
    train_labels = to_binary_categories(train_labels)
    true_labels = to_binary_categories(true_labels)
    true_labels = recreate_categories(true_labels)

    # c_range = np.arange(1, 10, 2)      # wide range
    c_range = np.arange(2, 16, 4)/10     # narrow range
    
    loss_c = []
    f1_score_c = []

    for c in c_range:
        print(bold(f'Model training - C={c}'))
        # fit model on training set
        hyperplane, l = fit(train_set, train_labels, C=c, learning_rate=1e-3, n_epoch=50)
        # predict on testing set
        pred_labels = predict(test_set, hyperplane)
        pred_labels = recreate_categories(pred_labels)
        
        loss_c.append(l)        

        f1_score_c.append(f1_score(pred_labels, true_labels, verbose=False))

    print(f1_score_c)

    # Plot the evolution of the loss through the iterations
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.plot()
    ax1 = plt.subplot(121)
    for i in range(len(c_range)):
        ax1.plot(np.arange(len(loss_c[i])), np.array(loss_c[i]), label=f'C={c_range[i]}')
    plt.title("Loss as a function of the number of epochs")
    plt.xlabel("Number of epochs"); plt.ylabel("Loss function")
    plt.legend()

    ax2 = plt.subplot(122)
    ax2.bar([f'C={c}' for c in c_range] , f1_score_c)
    plt.title("F-score as a function of $C$")
    plt.xlabel("Value of $C$"); plt.ylabel("F-score")
    plt.show()


if __name__ == "__main__":
    # --- Test of the SVM implementation
    run_SVM()

    # --- Feature importance
    # run_feature_importance()

    # --- Hyperparameter tuning
    # run_hyperparameter_tuning()
