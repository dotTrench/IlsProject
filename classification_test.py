from classifications.decisiontree import DecisionTree
from crossvalidation import read_all_datasets
import numpy as np



def test_decision_tree():
    datasets = read_all_datasets()
    dataset = datasets[2]

    filename = dataset[1]
    dataset = dataset[0]

    attributes = dataset.columns[:-1]
    d_class = dataset.columns[-1]

    features = dataset[list(attributes)]
    x = np.array(features.values)
    y = np.array(dataset[d_class].values)

    tree = DecisionTree()
    tree.fit(x, y)


if __name__ == '__main__':
    test_decision_tree()