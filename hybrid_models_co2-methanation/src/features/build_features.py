"""Feature Engineering."""

from sklearn import linear_model
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import (VarianceThreshold,
                                       RFECV)

###############################################################################
# Feature Reduction
###############################################################################


def dimensionality_reduction_PCA(X_std):
    """
    Extract Features with PCA.

    PCA projects the features onto the principle components. The motivation is
    to reduce the features dimensionalty while only losing a small amount of
    information.
    """
    # Create a PCA that will retain 98% of the variance
    pca = PCA(n_components=0.98)
    # Fit the PCA and transform the data
    X_std_pca = pca.fit_transform(X_std)
    # Show results
    print('Original number of features:', X_std.shape[1])
    print('Reduced number of features:', X_std_pca.shape[1])
    print('Explained ratio:', pca.explained_variance_ratio_)
    return X_std_pca


def dimensionality_reduction_kernel_PCA(X_std):
    """
    Extract Features with Kernel PCA.

    KPCA can reduce dimensionality while making data linearly separable.
    """
    # Create a KPCA that will retain 99% of the variance
    kpca = KernelPCA(kernel="rbf", gamma=15, n_components=4)
    # Fit the PCA and transform the data
    X_std_kpca = kpca.fit_transform(X_std)
    # Show results
    print('Original number of features:', X_std.shape[1])
    print('Reduced number of features:', X_std_kpca.shape[1])
    print('Explained ratio:', kpca.explained_variance_ratio_)
    return X_std_kpca

###############################################################################
# Feature Selection
###############################################################################


def feature_selection_by_variance_tresholding(X_std):
    """Drop features below variance treshold."""
    # Create VarianceThreshold object with a variance with a threshold of 0.5
    thresholder = VarianceThreshold(threshold=.5)
    # Conduct variance thresholding
    X_std_hv = thresholder.fit_transform(X_std)
    print('Original number of features:', X_std.shape[1])
    print('Reduced number of features:', X_std_hv.shape[1])
    return X_std_hv


def feature_selection_by_recursive_feature_elimination(X_std, y_std):
    """Drop features below variance treshold."""
    # Create a linear regression
    ols = linear_model.LinearRegression()
    # Create recursive feature eliminator that scores features by MSE
    rfecv = RFECV(estimator=ols, step=1, scoring='neg_mean_squared_error')
    # Fit recursive feature eliminator
    rfecv.fit(X_std, y_std)
    # Recursive feature elimination
    X_std_rfecv = rfecv.transform(X_std)
    print('Original number of features:', X_std.shape[1])
    print('Reduced number of features:', X_std_rfecv.shape[1])
    return X_std_rfecv
