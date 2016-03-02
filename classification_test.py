# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from classifications.decision_tree import DecisionTree
from pandas import read_csv
from pprint import pprint

# load the iris datasets
dataset = datasets.load_iris()

# dt = DecisionTreeClassifier()
# dt.fit(dataset.data, dataset.target)

tree = DecisionTree()
tree.fit(dataset.data, dataset.target)

pprint(tree._root)
