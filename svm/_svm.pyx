import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.cdivision(True)
def _optimize(
        self, np.ndarray[np.float_t, ndim=2] K,
        np.ndarray[np.float_t, ndim=2] X,
        np.ndarray[np.float_t, ndim=1] y,
        np.ndarray[np.float_t, ndim=1] dual_coef_,
        double C,
        int n_samples,
        object random_state,
        int numpasses,
        int maxiter):
    cdef int it = 0
    cdef int passes = 0
    cdef int alphas_changed
    cdef int i, j
    cdef double Ei, Ej, ai, aj, newai, newaj, eta, L, H

    cdef double b = 0.0
    while passes < numpasses and it < maxiter:
        alphas_changed = 0
        for i in range(n_samples):
            Ei = self._margins(self.support_vectors_[np.newaxis, i], dual_coef_, y, b)[0] - y[i]
            if ((y[i] * Ei < -self.tol and dual_coef_[i] < C) or
                    (y[i] * Ei > self.tol and dual_coef_[i] > 0)):
                # self.alphas[i] needs updating! Pick a j to update it with
                j = i
                while j == i:
                    j = random_state.randint(n_samples)
                Ej = self._margins(self.support_vectors_[np.newaxis, j], dual_coef_, y, b)[0] - y[j]

                # compute L and H bounds for j to ensure we're in [0 C]x[0 C] box
                ai = dual_coef_[i]
                aj = dual_coef_[j]
                if y[i] == y[j]:
                    L = max(0, ai + aj - C)
                    H = min(C, ai + aj)
                else:
                    L = max(0, aj - ai)
                    H = min(C, aj - ai + C)

                if abs(L - H) < 1e-4:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # compute new alpha[j] and clip it inside [0 C]x[0 C]
                # box then compute alpha[i] based on it.
                newaj = aj - y[j] * (Ei - Ej) / eta
                if newaj > H:
                    newaj = H
                if newaj < L:
                    newaj = L
                if abs(aj - newaj) < 1e-4:
                    continue
                dual_coef_[j] = newaj
                newai = ai + y[i] * y[j] * (aj - newaj)
                dual_coef_[i] = newai

                # update the bias term
                b1 = (
                    b - Ei - y[i] * (newai - ai) * K[i, i] -
                    y[j] * (newaj - aj) * K[i, j])
                b2 = (
                    b - Ej - y[i] * (newai - ai) * K[i, j] -
                    y[j] * (newaj - aj) * K[j, j])
                b = 0.5 * (b1 + b2)
                if newai > 0 and newai < C:
                    b = b1
                if newaj > 0 and newaj < C:
                    b = b2

                alphas_changed += 1

        it += 1
        self.intercept_ = b

        if alphas_changed == 0:
            passes += 1
        else:
            passes = 0

        if self.verbose >= 2 and maxiter % (it + 1) == 0:
            print("[SVM] Finished iteration %d" % it)
