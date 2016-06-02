"""Support vector machine (SVM).

This is a binary SVM and is trained using the SMO algorithm.
Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)

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
        numerical tolerance. Don't touch unless you're pro

    alphatol : float, optional (default: 1e-7)
        non-support vectors for space and time efficiency are truncated. To
        guarantee correct result set this to 0 to do no truncating. If you
        want to increase efficiency, experiment with setting this little
        higher, up to maybe 1e-4 or so.

    maxiter : int, optional (default: 10000)
        max number of iterations

    numpasses : int, optional (default: 10)
        how many passes over data with no change before we halt? Increase for
        more precision.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.
    """
    def __init__(self, C=1.0, kernel="linear", gamma=None, tol=1e-4,
                 alphatol=1e-7, maxiter=10000, numpasses=10,
                 random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.alphatol = alphatol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.random_state = random_state

    def fit(self, X, y):
        self.X = check_array(X)
        self.y = check_array(y, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        n_samples, n_features = X.shape

        self.kernel_args = {}
        if self.gamma is not None:
            self.kernel_args["gamma"] = self.gamma
        K = pairwise_kernels(X, metric=self.kernel, **self.kernel_args)

        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        self.usew_ = False

        it = 0
        passes = 0
        while passes < self.numpasses and it < self.maxiter:
            print(it)
            alphaChanged = 0
            for i in range(n_samples):
                Ei = self.margin_one(self.X[i]) - self.y[i]
                if ((self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                        (self.y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    # self.alphas[i] needs updating! Pick a j to update it with
                    j = i
                    while j == i:
                        j = random_state.randint(n_samples)
                    Ej = self.margin_one(self.X[j]) - self.y[j]

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
                        H = min(self.C, self.C + aj - ai)

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

        # if the user was using a linear kernel, lets also compute and store the
        # weights. This will speed up evaluations during testing time
        if self.kernel == "linear":
            # compute weights and store them
            self.w = np.zeros(n_features)
            for j in range(n_features):
                for i in range(n_samples):
                    self.w[j] += self.alphas[i] * self.y[i] * self.X[i, j]
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
            self.alphas_ = self.alphas[support_vectors]
            self.X = X[support_vectors]
            self.y = y[support_vectors]

    def margin_one(self, x):
        # x is an array of length D. Returns margin of given example
        # this is the core prediction function. All others are for convenience
        # mostly and end up calling this one somehow.
        f = self.b
        # if the linear kernel was used and w was computed and stored,
        # (i.e. the svm has fully finished training)
        # the internal class variable usew_ will be set to true.
        if self.usew_:
            # we can speed this up a lot by using the computed weights
            # we computed these during train(). This is significantly faster
            # than the version below
            for j in range(self.X.shape[1]):
                f += x[j] * self.w[j]
        else:
            for i in range(self.X.shape[0]):
                f += self.alphas[i] * self.y[i] * pairwise_kernels(np.atleast_2d(x), self.X[np.newaxis, i], metric=self.kernel, **self.kernel_args)
        return f

    def margins(self, X):
        X = check_array(X)
        n_samples, n_features = X.shape
        # go over support vectors and accumulate the prediction. 
        margins = np.empty(n_samples)
        for i in range(n_samples):
            margins[i] = self.margin_one(X[i])
        return margins

    def predict(self, X):
        m = self.margins(X)
        y = np.ones_like(m)
        y[m <= 0.0] = -1.0
        return y


if __name__ == "__main__":
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    y[y > 0.0] = 1.0
    y[y <= 0.0] = -1.0
    svm = SVM(kernel="rbf", gamma=1.0, maxiter=2000, tol=0.0, alphatol=0.0, numpasses=np.inf, random_state=0)
    svm.fit(X, y)
    y_pred = svm.predict(X)
    print y
    print y_pred
    print svm.margins(X)
