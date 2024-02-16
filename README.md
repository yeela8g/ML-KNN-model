# KNN model
* This program is a Python implementation of a K-Nearest Neighbors (KNN) classifier for the CIFAR-10 dataset. CIFAR-10 is a widely used dataset in computer vision, containing 60,000 32x32 color images across 10 classes.
* The KNN algorithm is employed to classify new data points based on the majority class of their K nearest neighbors in the training dataset. This program trains a KNN classifier using the CIFAR-10 training set and evaluates its performance on the test set.

## Features:

> Loading and Preprocessing CIFAR-10 Dataset: The program loads the CIFAR-10 dataset and performs basic preprocessing steps such as normalization and batching.

> Implementing KNN Classifier: the KNN algorithm was implemented from scratch, including functions to calculate Euclidean distance between data points and the main KNeighborsClassifier class for model fitting and prediction.

> Model Evaluation: The program evaluates the classifier's performance by calculating error rates, generating a confusion matrix, and displaying a classification report including precision, recall, and F1-score for each class.

> Choosing the Best K Value: A function is included to find the optimal K value by training multiple KNN models with different K values and selecting the one with the lowest error rate.

> Visualization: The program provides a visualization of the error rate vs. K plot to assist in selecting the best K value.

## How to Use:

Run the program in a Python environment.
The program automatically loads the CIFAR-10 dataset, trains the KNN classifier, evaluates its performance, and displays the results.
Optionally, adjust hyperparameters such as the range of K values to explore for finding the best K.

---

This program was developed as part of an assignment for an Introduction to Machine Learning course, focusing on building and evaluating machine learning models. It provides a hands-on implementation of a K-Nearest Neighbors (KNN) classifier for the CIFAR-10 dataset, demonstrating fundamental concepts in machine learning model development and evaluation. 
