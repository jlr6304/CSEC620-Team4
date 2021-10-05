import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def tune_number_of_components(dataset):
    """
    ## Plot the explained variance ratio as a function of the number of components

        `parameters`
            dataset: data on which to perform PCA

        `return`
            plt: plots of the explained variance ratio and cumulative explained variance ratio
    
    """

    n_features = dataset.shape[1]

    # Use of a scaler
    # scaler = StandardScaler(copy=False).fit(dataset)
    # scaler.transform(dataset)

    # Fit PCA on dataset
    pca = PCA(n_components=n_features)
    pca.fit(dataset)    

    # PLOTS OF THE VARIANCE RATIO
    plt.figure(1, (14, 7))
    # first plot
    ax1 = plt.subplot(121)
    ax1.bar(np.arange(n_features), pca.explained_variance_ratio_ * 100)
    plt.xlabel("number of components"); plt.ylabel("explained variance ratio (%)")
    
    # second plot
    ax2 = plt.subplot(122)
    ax2.bar(np.arange(n_features), np.cumsum(pca.explained_variance_ratio_)*100)
    plt.xlabel("number of components"); plt.ylabel("cumulative explained variance ratio (%)")
    plt.show()
    
    return plt


def reduce_dimensions(training, testing, n_components = 5):
    """
    ## Reduce the number of features by applying PCA

        `parameters`
            training: training set on which to compute new components
            testing: testing set on which to apply new components

        `return`
            projected_training: transformed training set with the new components
            projected_testing: transformed testing set with the new components
    
    """
    
    # ---- Fit PCA components on training set
    pca = PCA(n_components=n_components)
    pca.fit(training)

    # ---- Project sets on PCA fitted components
    projected_training = pca.transform(training)
    projected_testing = pca.transform(testing)

    return projected_training, projected_testing


def compare2D(training, testing_normal, testing_attack):
    """
    ## Comparison of the testing normal and testing attack datasets in the 2 first dimensions of the PCA feature space

        `parameters`
            training: training set on which to compute the two components
            testing_normal: testing normal set to project on those two components
            testing_attack: testing attack set to project on those two components

        `return`
            plt: Projections of the testing_normal and testing_attack sets colored with their labels

    """

    # ---- Fit PCA on training set
    pca = PCA(n_components=2)  
    pca.fit(training)  

    # ---- Projection of the new data into the 'principal components' space
    testing_normal = pca.transform(testing_normal)
    testing_attack = pca.transform(testing_attack)

    # ---- Plot of the projections
    plt.figure()
    # Plot each projection
    plt.scatter(testing_normal[:, 0], testing_normal[:, 1], c='b', alpha=.4, edgecolors='none', label="Normal")
    plt.scatter(testing_attack[:, 0], testing_attack[:, 1], c='r', alpha=.4, edgecolors='none', label="Attack")
    # Graphical changes
    plt.xlabel("Dimension 1"); plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()

    return plt


def compare3D(training, testing_normal, testing_attack):
    """
    ## Comparison of the testing normal and testing attack datasets in the 3 first dimensions of the PCA feature space

        `parameters`
            training: training set on which to compute the three components
            testing_normal: testing normal set to project on those three components
            testing_attack: testing attack set to project on those three components

        `return`
            plt: Projections of the testing_normal and testing_attack sets colored with their labels

    """

    # ---- Fit PCA on training set
    pca = PCA(n_components=3)  
    pca.fit(training)

    # ---- Projection of the new data into the 'principal components' space
    testing_normal = pca.transform(testing_normal)
    testing_attack = pca.transform(testing_attack)

    # ---- Plot of the projections
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Plot each projection
    ax.scatter(testing_normal[:, 0], testing_normal[:, 1], testing_normal[:, 2], c='b', alpha=.4, edgecolors='none', label="Normal")
    ax.scatter(testing_attack[:, 0], testing_attack[:, 1], testing_attack[:, 2], c='r', alpha=.4, edgecolors='none', label="Attack")
    # Graphical changes
    ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2"); ax.set_zlabel("Dimension 3")
    ax.view_init(elev=35., azim=155)
    plt.legend()
    plt.show()

    return plt



if __name__ == "__main__":

    # ---- Load data
    training = np.load("data/training_normal.npy")
    testing_normal = np.load("data/testing_normal.npy")
    testing_attack = np.load("data/testing_attack.npy")

    # ---- Comparison of the normal/attack in the projected space
    compare2D(training, testing_normal, testing_attack)
    # compare3D(training, testing_normal, testing_attack)

    # ---- Choose the best number of features for the dimension reduction
    # tune_number_of_components(training)

    # ---- Reduce the number of features (dimension reduction)
    # reduce_dimensions(training, testing_normal, n_components = 5)