import numpy as np
from svm import SVM
from sklearn.metrics import accuracy_score
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_greater


def test_linear_svm():
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 2)
    b = -0.2
    w = np.array([0.5, -0.3])
    y = np.sign(b + np.dot(X, w))

    svm = SVM(kernel="linear", C=1.0)
    svm.fit(X, y)
    y_pred = svm.predict(X)
    assert_array_almost_equal(y, y_pred)
    assert_true(hasattr(svm, "coef_"))
    assert_true(hasattr(svm, "intercept_"))


def test_rbf_svm():
    random_state = np.random.RandomState(0)
    n_samples = 100
    X = np.empty((n_samples, 2))
    X[:, 0] = np.linspace(0, 1, n_samples)
    X[:, 1] = random_state.randn(n_samples)
    y = np.sign(X[:, 1] - np.sin(2.0 * np.pi * np.sin(X[:, 0])))

    svm = SVM(kernel="rbf", C=1.0, gamma=1.0)
    svm.fit(X, y)
    y_pred = svm.predict(X)

    assert_greater(accuracy_score(y, y_pred), 0.9)


def test_poly_svm():
    random_state = np.random.RandomState(0)
    n_samples = 100
    X = np.empty((n_samples, 2))
    X[:, 0] = np.linspace(0, 1, n_samples)
    X[:, 1] = random_state.randn(n_samples)
    y = np.sign(X[:, 1] - np.sin(2.0 * np.pi * np.sin(X[:, 0])))

    svm = SVM(kernel="poly", C=1.0, degree=3, coef0=1.0)
    svm.fit(X, y)
    y_pred = svm.predict(X)

    assert_greater(accuracy_score(y, y_pred), 0.9)
