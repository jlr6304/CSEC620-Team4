# -- Import modules 
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# Personnalized metric: F1-score
def f1_m(y_true, y_pred):
    """
    ### Computes F1-score using Keras syntax
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def run_NN(X_train, y_train, X_test, y_test, activation='relu', n_nodes=200, n_epochs=200, batch_size=50):
    """
    ### Run Neural Network on the dataset and save the predicted labels as a file 
    
    The network is a dense 1 hidden layer NN 
    """
    # ---- Neural Network Initialization
    n = X_train.shape[1]

    model = Sequential()
    model.add(Dense(30,  input_dim=n, activation=activation))
    model.add(Dense(1,  activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

    # ---- Training
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
    
    # ---- Predict on testing
    y_pred = (model.predict(X_test)[:,0]>.5)*1
    _, f_score = model.evaluate(X_test, y_test)
    print('F-score: %.2f' % (f_score*100))

    np.save("data/pred_labels_NN.npy", y_pred)
    return y_pred


def tune_NN(X_train, y_train, activation=['sigmoid', 'softmax', 'relu'], n_nodes=[5, 10, 20, 30, 100, 200, 500], n_epochs=50, batch_size=50):
    """
    ### Tune Neural Network activation function and number of nodes
    
    The tuning is done on a validation set different from the testing set 
    """

    
    # splitting into train and validation set
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size = .3)

    # ---- Models training
    scores = np.zeros((len(activation),len(n_nodes)))
    for i,act in enumerate(activation):
        for j, num in enumerate(n_nodes):
            n = X_train.shape[1]
            
            # ---- Neural Network Initialization
            model = Sequential()
            model.add(Dense(num,  input_dim=n, activation=act))
            model.add(Dense(1,  activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

            # ---- Training
            history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
        
            # ---- Store score
            _, scores[i, j] = model.evaluate(X_vali, y_vali)
    

    df = pd.DataFrame(scores, index=activation, columns=n_nodes)
    sns.heatmap(df, annot=True, fmt='.3f')
    plt.title("F1-score")
    plt.xlabel("number of nodes"); plt.ylabel("activation function")
    plt.show()


def check_convergence(X_train, y_train, activation = 'relu', n_nodes=100, n_epochs=200, batch_size=50):
    """
    ### Displays the convergence of the F1-score with the epochs and depending on batch size 
    
    """
    # ---- Neural Network Initialization
    n = X_train.shape[1]
    
    model = Sequential()
    model.add(Dense(n_nodes,  input_dim=n, activation=activation))
    model.add(Dense(1,  activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

    # ---- Training
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)

    # ---- Plot convergence   
    plt.plot(history.history['f1_m'])
    plt.title('Model F-score evolution')
    plt.ylabel('F-score')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    

if __name__=='__main__':

    # -- Load data
    X_train = np.load("data/train_set.npy", allow_pickle=True)
    y_train = np.load("data/train_labels.npy", allow_pickle=True)

    X_test = np.load("data/test_set.npy", allow_pickle=True)
    y_test = np.load("data/test_labels.npy", allow_pickle=True)

    # -- Run Neural Network
    run_NN(X_train, y_train, X_test, y_test, activation='relu', n_nodes=200, n_epochs=200, batch_size=50)
    
    # -- Tune Neural Network
    # tune_NN(X_train, y_train)

    # -- Neural Network convergence
    # check_convergence(X_train, y_train, n_epochs=200, batch_size=50)