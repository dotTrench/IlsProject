from classifications.decisiontree import DecisionTree
from sklearn import datasets


def test_decision_tree():
    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    tree = DecisionTree()
    tree.fit(x, y)


