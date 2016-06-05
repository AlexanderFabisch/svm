#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=False
cimport numpy as np
import numpy as np


def smo(np.ndarray[np.float_t, ndim=2] K,
        np.ndarray[np.float_t, ndim=1] y,
        np.ndarray[np.float_t, ndim=1] dual_coef_,
        double C,
        object random_state,
        double tol,
        int numpasses,
        int maxiter,
        int verbose):
    cdef int n_samples = K.shape[0]
    cdef int it = 0
    cdef int passes = 0
    cdef int alphas_changed
    cdef int i, j
    cdef double Ei, Ej, ai, aj, newai, newaj, eta, L, H, b1, b2, current_y

    cdef np.ndarray[np.float_t, ndim=2] yK = y * K

    cdef double b = 0.0
    while passes < numpasses and it < maxiter:
        alphas_changed = 0
        for i in range(n_samples):
            current_y = _margins_kernel(yK[i], dual_coef_, b)
            Ei = current_y - y[i]
            if ((y[i] * Ei < -tol and dual_coef_[i] < C) or
                    (y[i] * Ei > tol and dual_coef_[i] > 0)):
                # self.alphas[i] needs updating! Pick a j to update it with
                j = i
                while j == i:
                    j = random_state.randint(n_samples)
                current_y = _margins_kernel(yK[j], dual_coef_, b)
                Ej = current_y - y[j]

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
                newaj = min(newaj, H)
                newaj = max(newaj, L)
                if abs(aj - newaj) < 1e-4:
                    continue
                dual_coef_[j] = newaj
                newai = ai + y[i] * y[j] * (aj - newaj)
                dual_coef_[i] = newai

                # update the bias term
                b1 = (b - Ei - y[i] * (newai - ai) * K[i, i] -
                      y[j] * (newaj - aj) * K[i, j])
                b2 = (b - Ej - y[i] * (newai - ai) * K[i, j] -
                      y[j] * (newaj - aj) * K[j, j])
                b = 0.5 * (b1 + b2)

                if newai > 0 and newai < C:
                    b = b1
                if newaj > 0 and newaj < C:
                    b = b2

                alphas_changed += 1

        it += 1

        if alphas_changed == 0:
            passes += 1
        else:
            passes = 0

        if verbose >= 2 and (it + 1) % (maxiter / 10) == 0:
            print("[SVM] Finished iteration %d" % it)

    return b


cdef double _margins_kernel(
        np.ndarray[np.float_t, ndim=1] yk,
        np.ndarray[np.float_t, ndim=1] dual_coef,
        double intercept):
    return intercept + np.dot(dual_coef, yk)