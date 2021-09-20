# List of tasks for next assignment

---
## Dimensionality Reduction

**Question 1. a)b)c)** 

- Realize PCA on training set

- merge subset of testing sets (normal and attacks) and scatter plot on those 2 dimensions.

- try to draw a boundary between attacks and normal 

- preprocess function to obtain train/test data for the clustering algorithms 

- Elbow method for selection of the components

---
## k-means

- Implement algorithm

- try and compare several hyperparameters: **Question 2.b)**
    - Number of k clusters to build.
    - Distance threshold value t
    - distance metric

- advantages and drawbacks of the algorithm for anomaly detection: **Question 3.a)**

- overfit/underfit issue based on the accuracy of training and testing dataset: **Question 4.b)**

---
## DBSCAN

- Implement algorithm

- try and compare several hyperparameters: **Question 2.b)**
    - The epsilon value used to determine neighbors.
    - The minimum number of neighbors to be labeled as a core point.
    - distance metric

- advantages and drawbacks of the algorithm for anomaly detection: **Question 3.a)**

- table/figure of the scores of the combinations of the two hyperparameters + disccusion: **Question 3.b)**

- overfit/underfit issue based on the accuracy of training and testing dataset: **Question 4. b)**

---
## Hyperparameter tuning 

- choose the method to provide a function for validation set: **Question 2. a)c)**

---
## Performance

- Test the detection rate (DR) and false alarm rate (FAR) and other performance metrics + discussion: **Question 4. a)c)**
