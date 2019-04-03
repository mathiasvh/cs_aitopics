import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing


data_file = "Affairs.csv"

df = pd.read_csv(data_file, delimiter=',',
                 usecols=['affairs','gender','age','yearsmarried','children',
                          'religiousness','education','occupation','rating'],
                 converters={'children': lambda s: True if s == "yes" else False})

def main():
    # ommit gender and children (as does the reference paper)
    X = df[["yearsmarried","age","religiousness","occupation","rating"]].values
    y = df['affairs'].values

    # shuffle data
    order = np.argsort(np.random.random(y.shape))
    X = X[order]
    y = y[order]


    names = ["yearsmarried","age","religiousness","occupation","rating"]

    print("features to be plotted on first graph: " + str(names))
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X, y)

    print('Convenience plot with ``partial_dependence_plots``')

    # features = [0, 5, 1, 2, (5, 1)]
    features = [0,1,2,3,4,(0,1)]
    fig, axs = plot_partial_dependence(clf, X, features,
                                       feature_names=names,
                                       n_jobs=-1, grid_resolution=100, n_cols=3)
    fig.set_size_inches(10.5, 7.5)
    fig.suptitle('Partial dependence for amount of affairs\n'
                 'for the Affairs dataset.')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap

    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()

    target_feature = (0, 1)
    pdp, axes = partial_dependence(clf, target_feature,
                                   X=X, grid_resolution=100)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].reshape(list(map(np.size, axes))).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of amount of affairs\n'
                 'for the amount of years married and the age.')
    plt.subplots_adjust(top=0.9)

    plt.show()


# Necessary on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
