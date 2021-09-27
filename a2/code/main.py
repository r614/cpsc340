#!/usr/bin/env python
import argparse
import collections
from functools import partial
import os
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.numeric import array_equal
from numpy.lib.function_base import append
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import validation

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
import utils
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
func_registry = {}


def handle(number):
    def register(func):
        func_registry[number] = func
        return func

    return register


def run(question):
    if question not in func_registry:
        raise ValueError(f"unknown question {question}")
    return func_registry[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True, choices=func_registry.keys())
    args = parser.parse_args()
    return run(args.question)


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [1, 3, 10]:
        classifier = KNN(k)
        classifier.fit(X, y)

        training_error = np.mean(classifier.predict(X) != y)

        y_pred = classifier.predict(X_test)
        test_error = np.mean(y_test != y_pred)

        print(f"K = {k}, training error: {training_error}, test error: {test_error}")

    # Generate Plots for SKlearn and custom KNN

    custom = KNN(1)
    custom.fit(X, y)
    sklearn = KNeighborsClassifier(n_neighbors=1)
    sklearn.fit(X, y)

    utils.plot_classifier(custom, X_test, y_test)
    fname = os.path.join("..", "figs", "q1_CustomKNN.png")
    plt.savefig(fname)
    print("\nSaved figure at '%s'" % fname)

    utils.plot_classifier(sklearn, X_test, y_test)
    fname = os.path.join("..", "figs", "q1_SklearnKNN.png")
    plt.savefig(fname)
    print("\nSaved figure at '%s'" % fname)


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    K = 10
    fold_size = len(X) // K

    masks = []

    start = 0
    end = start + fold_size

    while end < len(y):
        m = np.full(y.shape, False, dtype=bool)
        m[start:end] = True
        masks.append(m)
        start = end
        end += fold_size

    ks = list(range(1, 30, 4))

    cv_accs = []
    for k in ks:
        acc = []
        for m in masks:

            X_valid = X[m, :]
            Y_valid = y[m]

            X_train = X[~m]
            Y_train = y[~m]

            nn = KNeighborsClassifier(k)
            nn.fit(X_train, Y_train)

            validation_pred = nn.predict(X_valid)
            acc.append(np.mean(validation_pred == Y_valid))
        
        cv_accs.append(np.mean(acc))
    

    # Compute Accuracies for non-cv models
    test_acc = []
    training_acc = []
    for k in ks:
        nn = KNeighborsClassifier(k)
        nn.fit(X, y) 

        training_acc.append(np.mean(nn.predict(X) == y))
        test_acc.append(np.mean(nn.predict(X_test) == y_test))
    
    plt.figure()
    plt.plot(ks, cv_accs, label="cv_acc")
    plt.plot(ks, test_acc, label="test_acc")
    plt.title(f"Cross-Validation and Testing Accuracies v/s K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    
    fname = os.path.join("..", "figs", "q2_2_TestAcc.png")
    plt.savefig(fname)
    print("\nSaved figure at '%s'" % fname)

    plt.figure()
    plt.plot(ks, training_acc, label="train_acc")
    plt.title("Training Accuracies v/s K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    
    fname = os.path.join("..", "figs", "q2_2_TrainAcc.png")
    plt.savefig(fname)
    print("\nSaved figure at '%s'" % fname)




@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    raise NotImplementedError()


@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4"""
    raise NotImplementedError()


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE HERE FOR Q4"""
    raise NotImplementedError()


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1"""
    raise NotImplementedError()


@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
