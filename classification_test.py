# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from classifications.decision_tree import DecisionTree
from panads import read_csv

data = read_csv('dataset/')
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)

tree = DecisionTree()
tree.fit(dataset.data, dataset.target)
