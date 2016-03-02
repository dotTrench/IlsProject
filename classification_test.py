# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from classifications.decision_tree import DecisionTree
from pandas import read_csv
import pprint
# load the iris datasets
dataset = datasets.load_iris()

tree = DecisionTree()
t = tree.fit(dataset.data, dataset.target)
pprint.pprint(t)
