from sklearn import cross_validation, datasets
from classifications.decisiontree import DecisionTree
from classifications.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from os import listdir, path
from pandas import read_csv
from multiprocessing.dummy import Pool
from collections import Counter
import numpy as np
import time


class Timer:
    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()

    def get_seconds(self):
        return self.stop_time - self.start_time

    def get_milliseconds(self):
        return self.get_seconds() * 1000


def read_csv_file(path):
    return read_csv(path)


def read_all_datasets():
    multi_files = [path.join('dataset/multi', file) for file in listdir('dataset/multi')]
    bin_files = [path.join('dataset/binary', file) for file in listdir('dataset/binary')]
    all_files = bin_files + multi_files

    csv_files = [read_csv(f) for f in multi_files]
    # csv_files = [read_csv(f) for f in bin_files]
    return csv_files


def run_test_on_dataset(dataset):
    cv = cross_validation.KFold(n=10, n_folds=10)
    attributes = dataset.columns[:-1]
    d_class = dataset.columns[-1]
    # print(d_class)
    features = dataset[list(attributes)]

    # print(features)
    x = np.array(features.values)
    y = np.array(dataset[d_class].values)

    dtc = DecisionTreeClassifier()
    dt = DecisionTree()
    rfc = RandomForestClassifier()
    rf = RandomForest()
    t = Timer()
    print('Running tests')
    t.start()
    accuracy = cross_validation.cross_val_score(dt, x, y, cv=10, scoring='accuracy')
    precision = cross_validation.cross_val_score(dt, x, y, cv=10, scoring='precision_weighted')

    t.stop()

    print(t.get_milliseconds())
    # print(precision)
    print('Done')
    print(precision)
    print(accuracy)
    return {
        'accuracy':accuracy,
        'precision': precision
    }

def main():
    t = Timer()
    t.start()
    datasets = read_all_datasets()
    t.stop()
    print(t.get_milliseconds())
    # p = Pool()
    # p.map(run_test_on_dataset, datasets)
    run_test_on_dataset(datasets[2])
    # p.close()
    # p.join()

    # # m_datasets = read_all_datasets()
    # t.stop()
    # for dset in m_datasets:
    #     y = dset['data'].values
    #     print(y)
    #     break
    # y = [dataset['data'] .values[:, -1] for dataset in m_datasets]
    # print(y[13])

    # cv = cross_validation.KFold(n=3, n_folds=3)

    # iris = datasets.load_iris()
    # x, y = iris.data, iris.target

    # model = DecisionTree()
    # model2 = DecisionTreeClassifier()

    # classifiers = [{'classifier': model, 'name': 'DecisionTree'},
    #                {'classifier': model2, 'name': 'DecisionTreeClassifier'}]
    # scorings = ['recall', 'accuracy', 'precision', 'roc_auc']

    # scores = []

    # for c in classifiers:
    #     for dataset in m_datasets:
    #         recall = cross_validation.cross_val_score(c['classifier'], x, y, cv=cv,
    #                                                   scoring='recall')
    #         accuracy = cross_validation.cross_val_score(c['classifier'], x, y, cv=cv,
    #                                                     scoring='accuracy')
    #         precision = cross_validation.cross_val_score(c['classifier'], x, y, cv=cv,
    #                                                      scoring='precision')
    #         scores.append({
    #             'classifier_name': c['name'],
    #             'recall': recall,
    #             'accuracy': accuracy,
    #             'precision': precision
    #             })

if __name__ == '__main__':
    main()