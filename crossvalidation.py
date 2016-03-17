from sklearn import cross_validation, datasets
from classifications.decisiontree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
x, y = iris.data, iris.target

model = DecisionTree()

score = cross_validation.cross_val_score(model, x, y, scoring='log_loss')

print(score)
