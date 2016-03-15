from decision_tree_binary import DecisionTree
from random import shuffle, randint
import numpy as np
from sklearn import datasets


class RandomForest:
    def __init__(self, n_estimators=10):
        self.criterion = None
        self.max_features = None
        self.max_depth = None
        self.max_samples_leaf = None
        self.bagging = None
        self.sample_size = None

        self._decision_trees = []
        self.n_estimators = n_estimators

    def fit(self, x, y):

        # Create n random subsets from the x-list
        # Each tree is usualy trained on 2/3 of the data

        subsets_x, subsets_y = self._create_random_subsets(x, y)

        for i in range(self.n_estimators):
            tree = DecisionTree()
            tree.fit(subsets_x[i], subsets_y[i])
            self._decision_trees.append(tree)

    def _create_random_subsets(self, x, y):
        subsets_x = []
        subsets_y = []

        for i in range(self.n_estimators):
            subset_x = []
            subset_y = []
            num_samples = round(len(x) * 0.2)

            for j in range(num_samples):
                r = randint(0, len(x)-1)
                subset_x.append(x[r])
                subset_y.append(y[r])

            subsets_x.append(subset_x)
            subsets_y.append(subset_y)
        return np.array(subsets_x), np.array(subsets_y)

    def predict(self, x):
        # Predict the value in all the trees and it belongs to
        # the most frequent result.

        raise NotImplementedError('Not implemented')

    def predict_proba(self, x):
        raise NotImplementedError('Not implemented')


forest = RandomForest(500)
tree = DecisionTree()

iris = datasets.load_iris()
# tree.fit(iris.data, iris.target)
# tree.print_tree()
forest.fit(iris.data, iris.target)
# forest.fit([[3, 4], [2, 6], [4, 6], [1, 6], [9, 11], [21, 15]], ["T", "F", "T", "F", "T", "T"])
