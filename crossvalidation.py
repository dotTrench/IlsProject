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
from pprint import pprint
from scipy.stats import wilcoxon


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
    print(multi_files)
    csv_files = [(read_csv(f, header=None), f) for f in multi_files]
    # csv_files = [read_csv(f) for f in bin_files]
    return csv_files


def run_test_on_dataset(dataset):
    filename = dataset[1]
    dataset = dataset[0]

    attributes = dataset.columns[:-1]
    d_class = dataset.columns[-1]

    features = dataset[list(attributes)]
    x = np.array(features.values)
    y = np.array(dataset[d_class].values)
    dtc = DecisionTreeClassifier()
    dt = DecisionTree()
    rfc = RandomForestClassifier(n_estimators=25, max_features=len(x[0]), max_depth=18)
    rf = RandomForest(n_estimators=25, max_features=len(x[0]), max_depth=18)

    models = [(dtc, dt)]
    print('Running tests')
    results = []
    for m1, m2 in models:

        # Model 1
        name1 = str(type(m1))

        # print('Testing model: {0}'.format(name))
        t = Timer()
        t.start()
        print('recall', name1)
        recall = cross_validation.cross_val_score(m1, x, y, cv=10,
                                                  scoring='recall_weighted')
        # auc = cross_validation.cross_val_score(m, x, y, cv=10,
        #                                        scoring='roc_auc')

        print('accuracy', name1)
        accuracy = cross_validation.cross_val_score(m1, x, y, cv=10,
                                                    scoring='accuracy')
        print('precision', name1)
        prec = cross_validation.cross_val_score(m1, x, y, cv=10,
                                                scoring='precision_weighted')
        t.stop()
        result = {
            'recall': recall,
            # 'auc': auc,
            'accuracy': accuracy,
            'precision': prec,
            'time': t.get_milliseconds(),
            'model': name1,
            'dataset_name': filename
        }

        w1 = accuracy

        print(result)
        results.append(result)

        # Model 2
        name2 = str(type(m2))

        print('model: {0} done in {1:.5f}ms'.format(name2, t.get_milliseconds()))

        t = Timer()
        t.start()
        print('recall', name2)
        recall = cross_validation.cross_val_score(m2, x, y, cv=10,
                                                  scoring='recall_weighted')
        # auc = cross_validation.cross_val_score(m, x, y, cv=10,
        #                                        scoring='roc_auc')

        print('accuracy', name2)
        accuracy = cross_validation.cross_val_score(m2, x, y, cv=10,
                                                    scoring='accuracy')
        print('precision', name2)
        prec = cross_validation.cross_val_score(m2, x, y, cv=10,
                                                scoring='precision_weighted')
        t.stop()
        result = {
            'recall': recall,
            # 'auc': auc,
            'accuracy': accuracy,
            'precision': prec,
            'time': t.get_milliseconds(),
            'model': name2,
            'dataset_name': filename
        }
        print(result)
        results.append(result)
        print('model: {0} done in {1:.5f}ms'.format(name2, t.get_milliseconds()))

        w2 = accuracy

        r = wilcoxon(w1, w2)

        print(r)

    return results


def main():
    from pprint import pprint
    t = Timer()
    t.start()
    datasets = read_all_datasets()
    t.stop()
    print(t.get_milliseconds())
    # p = Pool()
    # results = p.map(run_test_on_dataset, datasets)
    results = run_test_on_dataset(datasets[1])
    print(results)
    # p.close()
    # p.join()

    # pprint(results)


if __name__ == '__main__':
    main()
