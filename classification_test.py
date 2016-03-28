from classifications.decisiontree import DecisionTree
from classifications.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from crossvalidation import read_all_datasets
import numpy as np


def test_decision_tree():
    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    tree = DecisionTree()
    tree.fit(x, y)


def test_random_forest():
    # Iris
    """
    iris = datasets.load_iris()

    x1 = iris.data
    y2 = iris.target
    """

    ds = read_all_datasets()
    dataset = ds[2]

    attributes = dataset.columns[:-1]
    d_class = dataset.columns[-1]

    features = dataset[list(attributes)]
    x = np.array(features.values)
    y = np.array(dataset[d_class].values)

    # print(len(x))
    # print(len(x[0]))
    # print(y)

    dt = DecisionTree()
    dt.fit(x, y)

    # dt = DecisionTreeClassifier()
    # dt.fit(x, y)

    # forest = RandomForest(n_estimators=1, max_depth=4, sample_size=0.1, max_features=4)
    # forest.fit(x, y)

if __name__ == '__main__':
    print("start")
    test_random_forest()

