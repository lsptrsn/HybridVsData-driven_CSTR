"""Visualize results."""
from sklearn_evaluation import plot
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve, learning_curve
from skopt.plots import plot_objective, plot_histogram

# https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve
# https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
# https://sklearn-evaluation.readthedocs.io/en/stable/user_guide/grid_search.html

# Load models
model_NuSVR = joblib.load('model_NuSVR.pkl')
model_GBR = joblib.load('model_GBR.pkl')
model_MLPR = joblib.load('model_MLPR.pkl')

# Partial Dependence plot of the objective function for regresor
_ = plot_objective(model_NuSVR.optimizer_results_[0])
# Plot of the histogram for LinearSVC
for i in range(len(model_MLPR.best_params_)):
    _ = plot_histogram(model_MLPR.optimizer_results_[0], i)
    plt.show()
pd.DataFrame(model_MLPR.cv_results_).plot(figsize=(10, 10))
plot.grid_search(model_MLPR.cv_results_,
                 change='regressor__algorithm__dropout_rate', kind='bar')

model = model_NuSVR
param_name = 'regressor__algorithm__C'
param_range = np.logspace(-3, 4, 5)
train_sizes = np.linspace(0.1, 1.0, 5)


def plot_validation_curve(X_val, y_val, model, param_name, param_range):
    """Calculate and lot the learning curve."""
    train_scores, test_scores = validation_curve(
        model.best_estimator_, X_val, y_val,
        param_name=param_name, param_range=param_range)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score",
        color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score",
        color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(X_test, y_test, model, train_sizes):
    """Calculate and lot the learning curve."""
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title('Learning Curve')
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model.best_estimator_, X_test, y_test, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

# Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g",
        label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt
