# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np  # this comes with Anaconda
import pandas as pd  # this comes with Anaconda
import matplotlib.pyplot as plt  # this comes with Anaconda
from scipy.optimize import approx_fprime  # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier  # if using Anaconda, install with `conda install scikit-learn`

""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

# CPSC 340 code
import utils
import grads
from decision_stump import DecisionStumpEquality, DecisionStumpErrorRate, DecisionStumpInfoGain
from decision_tree import DecisionTree


def main(question):
    if question == "3.4":
        # Here is some code to test your answers to Q3.4
        # Below we test out example_grad using scipy.optimize.approx_fprime, which approximates gradients.
        # if you want, you can use this to test out your foo_grad and bar_grad

        def check_grad(fun, grad):
            x0 = np.random.rand(5)  # take a random x-vector just for testing
            diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
            print("\n** %s **" % fun.__name__)
            print("My gradient     : %s" % grad(x0))
            print("Scipy's gradient: %s" % diff)
            print("\n\n")


        check_grad(grads.example, grads.example_grad)
        check_grad(grads.foo, grads.foo_grad)
        check_grad(grads.bar, grads.bar_grad)


    elif question == "5.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..', 'data', 'fluTrends.csv'))
        X = df.values

        print(df.describe())

        print ("\n")

        means = sorted([(col, df[col].mean()) for col in df], key=lambda x: x[1])
        variances = sorted([(col, df[col].var()) for col in df], key=lambda x: x[1])

        print(f"Lowest Mean: {means[0][0]}, {means[0][1]}")
        print(f"Highest Mean: {means[-1][0]}, {means[-1][1]}")
        print(f"Lowest variance: {variances[0][0]}, {variances[0][1]}")
        print(f"Highest variance: {variances[-1][0]}, {variances[-1][1]}")



    elif question == "6":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with equality rule error: %.3f"
              % error)

        # PLOT RESULT
        utils.plot_classifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.2":
        # Q2.2 - Decision Stump

        # 1. Load citiesSmall dataset
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plot_classifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.3":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with info gain rule error: %.3f" % error)

        # PLOT RESULT
        utils.plot_classifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.4":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2, stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

        utils.plot_classifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


        def print_stump(stump):
            print("Splitting on feature %s at threshold %f. >: %s, <=: %s" % (
            stump.j_best, stump.t_best, stump.y_hat_yes, stump.y_hat_no))


    elif question == "6.5":
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1, 15)  # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            # model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpEquality)
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time() - t))

        plt.plot(depths, my_tree_errors, label="errorrate")

        t = time.time()
        my_tree_errors_infogain = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth, stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time() - t))

        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time() - t))

        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q6_5_tree_errors.pdf")
        plt.savefig(fname)

        # plot the depth 15 sklearn classifier
        model = DecisionTreeClassifier(max_depth=15, criterion='entropy', random_state=1)
        model.fit(X, y)
        utils.plot_classifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_5_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    else:
        print("No code to run for question", question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    main(question)
