from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

cats = np.unique(y)
print(cats)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(X_train[:3])
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std[:3])

# ppn = Perceptron(
#     max_iter=40,
#     # to1=1e-8,
#     eta0=0.1,
#     random_state=0
# )
# ppn.fit(X_train_std, y_train)


def run_tests(classifier):
    y_test_pred = classifier.predict(X_test_std)
    print('Misclassified samples: %d, accuracy: %.3f' % (
        (y_test != y_test_pred).sum(),
        accuracy_score(y_test, y_test_pred))
    )
    for i, _ in enumerate(y_test):
        if (y_test[i] != y_test_pred[i]):
            print('X: {}, y_pred: {}, y_actual: {}'.format(X_test[i], y_test_pred[i], y_test[i]))


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy import ndarray


def min_max(X2d:ndarray) -> Tuple[float, float]:
    return X2d.min() - 1, X2d.max() + 1

def plot_decision_reg(
        X:ndarray,
        y:ndarray,
        classifier:Perceptron,
        test_idx=None,
        resolution:float=0.1):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot
    X0 = X[:, 0]
    X1 = X[:, 1]
    x0_min, x0_max = min_max(X0)
    x1_min, m1_max = min_max(X1)
    xx1, xx2, = np.meshgrid(
        np.arange(x0_min, x0_max, resolution),
        np.arange(x1_min, m1_max, resolution)
    )
    Z = classifier.predict(
        np.array([
            xx1.ravel(),
            xx2.ravel()
        ]).T
    )
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y==cl, 0],
            y=X[y==cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker=markers[idx],
            label=cl
        )
    if test_idx:
        Xt, yt = X[test_idx, :], y[test_idx]
        plt.scatter(
            x=Xt[:, 0],
            y=Xt[:, 1],
            alpha=1.0,
            c='',
            edgecolors='black',
            linewidths=1,
            marker='o',
            s=55,
            label='test set'
        )


X_all = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))


def plot_all(classifier):
    plot_decision_reg(
        X=X_all,
        y=y_all,
        classifier=classifier,
        test_idx=range(105, 150)
    )
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend(loc='upper left')
    plt.show()


# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=0.1, random_state=0)
# lr.fit(X_train_std, y_train)
# print(lr.coef_[1])
# plot_all(lr)
# plot_all(ppn)

# from sklearn.svm import SVC
# svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm.fit(X_train_std, y_train)
# run_tests(svm)
# plot_all(svm)

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log')
sgd.fit(X_train_std, y_train)
run_tests(sgd)
plot_all(sgd)
