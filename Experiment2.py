from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from classifications.decisiontree import DecisionTree
from classifications.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
from os import listdir, path
from multiprocessing.dummy import Pool
import numpy as np


def read_csv_file(path):
    return read_csv(path)


def read_all_datasets():
    multi_files = [path.join('dataset/multi', file) for file in listdir('dataset/multi')]
    bin_files = [path.join('dataset/binary', file) for file in listdir('dataset/binary')]
    all_files = bin_files + multi_files

    csv_files = [read_csv(f) for f in multi_files]
    # csv_files = [read_csv(f) for f in bin_files]
    return csv_files


def experim():
    ds = read_all_datasets()
    experiment2(ds[0])
    # p = Pool()
    # p.map(experiment2, datasets[1])

    # p.close()
    # p.join()


def experiment2(dataset):
    # Adjust the data
    attributes = dataset.columns[:-1]
    d_class = dataset.columns[-1]
    # print(d_class)
    features = dataset[list(attributes)]

    # print(features)
    x = np.array(features.values)
    y = np.array(dataset[d_class].values)

    # Fix one param_grid for each algorithm (Er beslutsträd algoritm, er Random Forest-implementering
    # DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier
    # Create params for all the algorithms with the parameters that should be optimised
    print(len(x[1]))
    max_features = list(range(1, len(x[0])))
    max_depth = list(range(1, len(x[0])))
    n_estimators = list(range(1, 50))
    n_neighbors = list(range(1, 50))

    param_grid_dt = dict(max_features=max_features, max_depth=max_depth)

    # param_grid_rf = dict(n_estimators=n_estimators, max_features=k_range)

    param_grid_dtc = dict(max_features=max_features, max_depth=max_depth)

    param_grid_rfc = dict(n_estimators=n_estimators, max_features=max_features)

    param_grid_knn = dict(n_neighbors=n_neighbors)

    # Add all param_grids to a single list
    param_grids = [param_grid_dt, param_grid_dtc, param_grid_rfc, param_grid_knn]

    # Add all the algorithms into a list of models
    models = [DecisionTree(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
    # RandomForest, DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

    scores = ['accuracy', 'precision_weighted', 'recall_weighted']

    for i in range(len(models)):
        for score in scores:
            print(type(models[i]))
            # Set n_jobs = -1 maybe?
            grid = GridSearchCV(models[i], param_grids[i], cv=10, scoring=score)
            grid.fit(x, y)
            print(score + " " + str(grid.best_params_))
            print()

if __name__ == '__main__':
    # experiment2()
    experim()
