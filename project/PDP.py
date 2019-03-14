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

train_percentage = 0.8

def main():
    # ommit gender and children (as does the reference paper)
    X = df[["yearsmarried","age","religiousness","occupation","rating"]].values
    y = df['affairs'].values

    # shuffle data
    order = np.argsort(np.random.random(y.shape))
    X = X[order]
    y = y[order]

    # split train/test
    cutoff_idx = int(train_percentage * y.shape[0])
    X_train = X[0:cutoff_idx, :]
    X_test  = X[cutoff_idx:, :]
    y_train = y[0:cutoff_idx]
    y_test  = y[cutoff_idx:]


    names = ["yearsmarried","age","religiousness","occupation","rating"]

    print(names)
    print("Training GBRT...")
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)
    print(" done.")

    print('Convenience plot with ``partial_dependence_plots``')

    # features = [0, 5, 1, 2, (5, 1)]
    features = [0,1,2,3,4,(0,1)]
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=50)
    fig.suptitle('Partial dependence for amount of affairs\n'
                 'for the Affairs dataset.')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()

    target_feature = (0, 1)
    pdp, axes = partial_dependence(clf, target_feature,
                                   X=X_train, grid_resolution=50)
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
                 ' for the amount of years married and the age.')
    plt.subplots_adjust(top=0.9)

    plt.show()


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
