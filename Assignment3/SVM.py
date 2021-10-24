import os

import numpy as np
from numpy.lib.shape_base import expand_dims
import torch
from functions import import_data, split_train_test, to_binary_categories, recreate_categories, confusion_matrix, f1_score, bold

import matplotlib.pyplot as plt

def fit(train_set, train_labels, C=5, learning_rate=.001, n_epoch=10):
    """
    #### Fits an hyperplane to the training set 
    parameters: 
        - train_set: set of samples to separate with an hyperplane
        - train_labels: labels of the previous samples in {-1, 1}
        - C: classification error penalizing coefficient
    return:
        - hyperplane: the hyperplane of the minimized model
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

    # Loop on epochs
    for e in range(n_epoch):
        print(f'Epoch {e+1}/{n_epoch}')
        
        # Loop on samples
        for i in range(train_set.shape[0]):

            y = torch.matmul(x, w) - b.expand_as(t)
            # Hinge Loss
            HL = torch.maximum( # element-wise maximum
                ones - torch.mul(t,y),
                zeros
            ).sum()
            
            # Classification loss
            CL = C*torch.norm(w)/2 # Frobenius norm is equivalent to euclidian norm for an array
            
            # Loss function and backward
            loss = HL + CL

            loss.backward()

            # Update slope and interset
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad.zero_()
            b.grad.zero_()

        loss_evol.append(loss.item())

    # Convert tensors to np.arrays and float
    # w = w.detach().numpy() 
    # b = b.detach().float()

    return {'slope':w, 'intercept':b}, loss_evol


def predict(set, hyperplane):
    """
    #### Predict the labels of the set samples based on the SVM hyperplane 
    parameters:
        - test_set: set of samples to categorize
        - hyperplane: SVM hyperplane to perform the classification
    """
    n_samples = set.shape[0]
    n_features = set.shape[1]
    
    # slope    
    w = hyperplane['slope']
    # intercept
    b = hyperplane['intercept'].expand(n_samples)

    # print(set.shape)
    # print(w.shape)
    # print(b.shape)

    set = torch.from_numpy(set).type(torch.float)

    labels = (torch.matmul(set, w) - b).sign()

    # Convert to a numpy array and return 
    labels = labels.detach().numpy().astype(int)
    return labels

def test_SVM():
    # import data from the preprocessed dataset
    df, labels = import_data()
    
    # split into training and testing sets
    train_set, train_labels, test_set, test_labels = split_train_test(df, labels)

    # convert labels into -1 and 1
    train_labels = to_binary_categories(train_labels)
    test_labels = to_binary_categories(test_labels)

    # fit model on training set
    hyperplane, l = fit(train_set, train_labels, n_epoch=50)

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

    # Plot the evolution of the loss through the iterations
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.plot(np.arange(len(l)), np.array(l))
    plt.title("Loss as a function of the number of epochs")
    plt.xlabel("Number of epochs"); plt.ylabel("Loss function") 
    plt.show()

if __name__ == "__main__":
    test_SVM()