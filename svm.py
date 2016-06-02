"""Support vector machine (SVM).


Simple usage example:
svm = SVM();
svm.fit(data, labels);
testlabels = svm.predict(testdata);
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_random_state


class SVM(BaseEstimator):
    """Support vector machine (SVM).

    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm"
    (http://math.unt.edu/~hsp0009/smo.pdf)

    Parameters
    ----------
    C : float, optional (default: 1)
        C value. Decrease for more regularization

    kernel : string, optional (default: 'linear')
        Kernel function, valid values are
            ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

    gamma : float, optional (default: None)
        Parameter for RBF kernel

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
    """
    def __init__(self, C=1.0, kernel="linear", gamma=None, tol=1e-4,
                 alphatol=1e-7, maxiter=10000, numpasses=10,
                 random_state=None, verbose=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.alphatol = alphatol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.X = check_array(X)
        self.y = check_array(y, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        n_samples, n_features = X.shape

        self.kernel_args = {}
        if self.kernel == "rbf" and self.gamma is not None:
            self.kernel_args["gamma"] = self.gamma
        K = pairwise_kernels(X, metric=self.kernel, **self.kernel_args)

        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        self.usew_ = False

        it = 0
        passes = 0
        while passes < self.numpasses and it < self.maxiter:
            alphaChanged = 0
            for i in range(n_samples):
                Ei = self.margins(self.X[np.newaxis, i])[0] - self.y[i]
                if ((self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                        (self.y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    # self.alphas[i] needs updating! Pick a j to update it with
                    j = i
                    while j == i:
                        j = random_state.randint(n_samples)
                    Ej = self.margins(self.X[np.newaxis, j])[0] - self.y[j]

                    # calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
                    ai = self.alphas[i]
                    aj = self.alphas[j]
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
                    self.alphas[j] = newaj
                    newai = ai + self.y[i] * self.y[j] * (aj - newaj)
                    self.alphas[i] = newai

                    # update the bias term
                    b1 = (self.b - Ei - self.y[i] * (newai - ai) * K[i, i] -
                          self.y[j] * (newaj - aj) * K[i, j])
                    b2 = (self.b - Ej - self.y[i] * (newai - ai) * K[i, j] -
                          self.y[j] * (newaj - aj) * K[j, j])
                    b = 0.5 * (b1 + b2)
                    if newai > 0 and newai < self.C:
                        self.b = b1
                    if newaj > 0 and newaj < self.C:
                        self.b = b2

                    alphaChanged += 1

            it += 1

            if alphaChanged == 0:
                passes += 1
            else:
                passes = 0

            if self.verbose >= 2 and self.maxiter % (it + 1) == 0:
                print("[SVM] Finished iteration %d" % it)

        # if the user was using a linear kernel, lets also compute and store the
        # weights. This will speed up evaluations during testing time
        if self.kernel == "linear":
            # compute weights and store them
            self.w = np.zeros(n_features)
            for j in range(n_features):
                self.w[j] += np.dot(self.alphas * self.y, self.X[:, j])
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
            support_vectors = np.nonzero(self.alphas)
            self.n_support_vectors = len(support_vectors[0])
            self.alphas = self.alphas[support_vectors]
            self.X = X[support_vectors]
            self.y = y[support_vectors]

    def margins(self, X):
        X = check_array(X)
        n_samples, n_features = X.shape

        if self.usew_:
            y = np.empty(n_samples)
            for n in range(n_samples):
                y[n] = float(self.w.dot(X[n]) + self.b)
        else:
            K = pairwise_kernels(X, self.X, metric=self.kernel,
                                 **self.kernel_args)
            y = self.b + np.sum(self.alphas[np.newaxis, :] * self.y * K, axis=1)

        return y

    def predict(self, X):
        return np.sign(self.margins(X))


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    random_state = np.random.RandomState(0)
    X = random_state.randn(100, 3)
    y = random_state.randn(100)
    y[y > 0.0] = 1.0
    y[y <= 0.0] = -1.0
    svm = SVM(kernel="rbf", gamma=10.0, random_state=0)
    svm.fit(X, y)
    print(accuracy_score(y, svm.predict(X)))
    print(svm.margins(X))
