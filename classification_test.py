# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from classifications.decision_tree import DecisionTree
from classifications.decision_tree_binary import *
from pandas import read_csv
from pprint import pprint
# load the iris datasets
dataset = datasets.load_iris()

# dt = DecisionTreeClassifier()
# dt.fit(dataset.data, dataset.target)
g_calc(np.array([[3, 3], [2, 1], [1, 9], [4, 110]]), np.array(['T', 'F', 'F', 'T']))
# v = gini_calc([3, 2, 1, 4], ['T', 'F', 'F', 'T'])
# print(v)
# binTree = DecisionTreeBinary()
# binTree.fit(dataset.data, dataset.target)
