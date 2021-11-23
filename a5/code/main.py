#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from collections import defaultdict
from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LeastSquaresLossL2,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
import utils
from utils import load_dataset, load_trainval, load_and_split


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question", required=True, choices=sorted(_funcs.keys()) + ["all"]
    )
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(lr_model, X_train, y_train)
    utils.savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # kernel logistic regression with a polynomial kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    kernel = PolynomialKernel(2)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegPolynomial.png", fig)

    # kernel logistic regression with a RBF kernel
    kernel = GaussianRBFKernel(0.5)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegRBF.png", fig)

@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])
    optimizer = GradientDescentLineSearch()
    
    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    best_train_err = float('inf')
    best_train_params = None

    best_val_err = float('inf')
    best_val_params = None

    for i, sig in enumerate(sigmas):
        for j, lam in enumerate(lammys): 
            kernel = GaussianRBFKernel(sig)
            loss_fn = KernelLogisticRegressionLossL2(lam)
            klr_model = KernelClassifier(loss_fn, optimizer, kernel)
            klr_model.fit(X_train, y_train)

            train_err = np.mean(klr_model.predict(X_train) != y_train)
            val_err = np.mean(klr_model.predict(X_val) != y_val)

            train_errs[i][j] = train_err
            val_errs[i][j] = val_err

            if train_err < best_train_err:
                best_train_err = train_err
                best_train_params = (sig, lam)
            
            if val_err < best_val_err:
                best_val_err = val_err
                best_val_params = (sig, lam)

    print(f"Best training error: {best_train_err:.1%}")
    print(f"Best training params: sigma = {best_train_params[0]}, lammy = {best_train_params[1]}")
    
    loss_fn = KernelLogisticRegressionLossL2(best_train_params[1])
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_train_params[0])
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("best_train_grid_search.png", fig)

    print(f"Best validation error: {best_val_err:.1%}")
    print(f"Best validation params: sigma = {best_val_params[0]}, lammy = {best_val_params[1]}")


    loss_fn = KernelLogisticRegressionLossL2(best_val_params[1])
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_val_params[0])
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("best_val_grid_search.png", fig)

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    utils.savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    utils.savefig("animals_matrix.png", fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    utils.savefig("animals_random.png", fig)

    encoder = PCAEncoder(2)
    encoder.fit(X_train_standardized)
    Z = encoder.encode(X_train_standardized)

    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in random_is:
        xy = Z[i, [0, 1]]
        ax.annotate(animal_names[i], xy=xy)
    utils.savefig("animals_PCA.png", fig)

    abs_w = np.abs(encoder.W)
    print(f"First component max : {trait_names[np.argmax(abs_w[0])]}")
    print(f"Second component max: {trait_names[np.argmax(abs_w[1])]}")


@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    utils.savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    for batch_size in [1, 10, 100]:
        print(f"Batch Size: {batch_size}")
        optimizer = StochasticGradient(GradientDescent(), ConstantLR(0.0003), batch_size, max_evals=10)
        loss_fn = LeastSquaresLoss()
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)
        print(model.fs)  # ~700 seems to be the global minimum.

        print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
        print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

        print("----\n")

@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    learning_rates = {
        "ConstantLR": ConstantLR, 
        "InverseLR": InverseLR, 
        "InverseSquaredLR": InverseSquaredLR, 
        "InverseSqrtLR": InverseSqrtLR
    }

    fs = defaultdict()

    for name, learning_rate_getter in learning_rates.items(): 
        optimizer = StochasticGradient(GradientDescent(), learning_rate_getter(0.1), 10)
        loss_fn = LeastSquaresLoss()
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)

        fs[name] = model.fs
    
    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    for name in fs: 
        ax.plot(fs[name], label=name)
    ax.legend()

    utils.savefig("q4_3_learning_rates.png", fig)

if __name__ == "__main__":
    main()
