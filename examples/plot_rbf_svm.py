import numpy as np
import matplotlib.pyplot as plt
from svm import SVM


random_state = np.random.RandomState(0)
n_samples = 20
X = random_state.rand(n_samples, 2)
y = np.ones(n_samples)
y[X[:, 0] + 0.1 * random_state.randn(n_samples) < 0.5] = -1.0


plt.figure()
for i, C in enumerate([1.0, 1e1, 1e2, np.inf]):
    svm = SVM(kernel="rbf", C=C, gamma=10.0, random_state=random_state)
    svm.fit(X, y)

    X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    X_test = np.vstack(map(np.ravel, (X_grid, Y_grid))).T
    Z_grid = svm.predict(X_test).reshape(X_grid.shape)

    plt.subplot(2, 2, 1 + i)
    plt.contourf(X_grid, Y_grid, Z_grid, alpha=0.3)
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                c="green", s=100)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("$C = %g$" % C)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(())
    plt.yticks(())
plt.show()
