import numpy as np
import matplotlib.pyplot as plt
from svm import SVM


random_state = np.random.RandomState(0)
n_samples = 20
X = random_state.rand(n_samples, 2)
y = np.ones(n_samples)
y[X[:, 0] + 0.1 * random_state.randn(n_samples) < 0.5] = -1.0


plt.figure()
for i, C in enumerate([1e-3, 1.0, 1e3, np.inf]):
    svm = SVM(kernel="linear", C=C, random_state=random_state)
    svm.fit(X, y)

    xx = np.linspace(0, 1)
    a = -svm.coef_[0] / svm.coef_[1]
    yy = a * xx - svm.intercept_ / svm.coef_[1]

    plt.subplot(2, 2, 1 + i)
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                c="green", s=100)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(xx, yy, 'k-')
    plt.title("$C = %g$" % C)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(())
    plt.yticks(())
plt.show()
