import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_random_state


class SVM(BaseEstimator):
    """Support vector machine (SVM).

    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm"
    (http://math.unt.edu/~hsp0009/smo.pdf)
    Based on Karpathy's svm.js implementation:
    https://github.com/karpathy/svmjs

    Parameters
    ----------
    C : float, optional (default: 1)
        Penalty parameter C of the error term.

    kernel : string, optional (default: 'rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``

    degree : int, optional (default: 3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default: 1)
        Parameter for RBF kernel

    coef0 : float, optional (default: 0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, optional (default: 1e-4)
        Numerical tolerance. Usually this should not be modified.

    alphatol : float, optional (default: 1e-7)
        Non-support vectors for space and time efficiency are truncated. To
        guarantee correct result set this to 0 to do no truncating. If you
        want to increase efficiency, experiment with setting this little
        higher, up to maybe 1e-4 or so.

    maxiter : int, optional (default: 10000)
        Maximum number of iterations

    numpasses : int, optional (default: 10)
        How many passes over data with no change before we halt? Increase for
        more precision.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    support_vectors_ : array-like, shape = [n_support_vectors, n_features]
        Support vectors.

    coef_ : array, shape = [n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : float
        Constant in decision function.
    """
    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma=1.0, coef0=0.0,
                 tol=1e-4, alphatol=1e-7, maxiter=10000, numpasses=10,
                 random_state=None, verbose=0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.alphatol = alphatol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.support_vectors_ = check_array(X)
        self.y = check_array(y, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        n_samples, n_features = X.shape

        self.kernel_args = {}
        if self.kernel == "rbf" and self.gamma is not None:
            self.kernel_args["gamma"] = self.gamma
        elif self.kernel == "poly":
            self.kernel_args["degree"] = self.degree
            self.kernel_args["coef0"] = self.coef0
        elif self.kernel == "sigmoid":
            self.kernel_args["coef0"] = self.coef0

        K = pairwise_kernels(X, metric=self.kernel, **self.kernel_args)

        self.dual_coef_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        self.usew_ = False

        it = 0
        passes = 0
        while passes < self.numpasses and it < self.maxiter:
            alphas_changed = 0
            for i in range(n_samples):
                Ei = self.margins(self.support_vectors_[np.newaxis, i])[0] - self.y[i]
                if ((self.y[i] * Ei < -self.tol and self.dual_coef_[i] < self.C) or
                        (self.y[i] * Ei > self.tol and self.dual_coef_[i] > 0)):
                    # self.alphas[i] needs updating! Pick a j to update it with
                    j = i
                    while j == i:
                        j = random_state.randint(n_samples)
                    Ej = self.margins(self.support_vectors_[np.newaxis, j])[0] - self.y[j]

                    # calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
                    ai = self.dual_coef_[i]
                    aj = self.dual_coef_[j]
                    L = 0
                    H = self.C
                    if y[i] == y[j]:
                        L = max(0, ai + aj - self.C)
                        H = min(self.C, ai + aj)
                    else:
                        L = max(0, aj - ai)
                        H = min(self.C, aj - ai + self.C)

                    if abs(L - H) < 1e-4:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # compute new self.alphas[j] and clip it inside [0 C]x[0 C] box
                    # then compute self.alphas[i] based on it.
                    newaj = aj - y[j] * (Ei - Ej) / eta
                    if newaj > H:
                        newaj = H
                    if newaj < L:
                        newaj = L
                    if abs(aj - newaj) < 1e-4:
                        continue
                    self.dual_coef_[j] = newaj
                    newai = ai + self.y[i] * self.y[j] * (aj - newaj)
                    self.dual_coef_[i] = newai

                    # update the bias term
                    b1 = (self.intercept_ - Ei - self.y[i] * (newai - ai) * K[i, i] -
                          self.y[j] * (newaj - aj) * K[i, j])
                    b2 = (self.intercept_ - Ej - self.y[i] * (newai - ai) * K[i, j] -
                          self.y[j] * (newaj - aj) * K[j, j])
                    b = 0.5 * (b1 + b2)
                    if newai > 0 and newai < self.C:
                        self.intercept_ = b1
                    if newaj > 0 and newaj < self.C:
                        self.intercept_ = b2

                    alphas_changed += 1

            it += 1

            if alphas_changed == 0:
                passes += 1
            else:
                passes = 0

            if self.verbose >= 2 and self.maxiter % (it + 1) == 0:
                print("[SVM] Finished iteration %d" % it)

        # if the user was using a linear kernel, lets also compute and store the
        # weights. This will speed up evaluations during testing time
        if self.kernel == "linear":
            # compute weights and store them
            self.coef_ = np.zeros(n_features)
            for j in range(n_features):
                self.coef_[j] += np.dot(self.dual_coef_ * self.y, self.support_vectors_[:, j])
            self.usew_ = True
        else:
            self.usew_ = False

            # okay, we need to retain all the support vectors in the training
            # data, we can't just get away with computing the weights and
            # throwing it out
            # But! We only need to store the support vectors for evaluation of
            # testing instances. So filter here based on self.self.alphas[i]. The
            # training data for which self.self.alphas[i] = 0 is irrelevant for
            # future. 
        support_vectors = np.nonzero(self.dual_coef_)
        self.dual_coef_ = self.dual_coef_[support_vectors]
        self.support_vectors_ = X[support_vectors]
        self.y = y[support_vectors]

    def margins(self, X):
        X = check_array(X)

        if self.usew_:
            y = np.dot(X, self.coef_) + self.intercept_
        else:
            K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel,
                                 **self.kernel_args)
            y = self.intercept_ + np.sum(self.dual_coef_[np.newaxis, :] * self.y * K, axis=1)

        return y

    def predict(self, X):
        return np.sign(self.margins(X))
