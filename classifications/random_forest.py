from classifications.decision_tree_binary import DecisionTree
from random import shuffle
import numpy as np
from sklearn import datasets

class RandomForest:
    def __init__(self, n_estimators):
        self.criterion = None
        self.max_features = None
        self.max_depth = None
        self.max_samples_leaf = None
        self.n_estimators = n_estimators # Nr of trees
        self.bagging = None # Bootstrap Aggregating
        self.sample_size = None # size of bootstrap samples (0, 1)

    def fit(self, x, y):

        # Create n random subsets from the x-list
        # Each tree is usualy trained on 2/3 of the data
        subsets_x, subsets_y = self._create_random_subsets(x, y)

        # Create n decision trees from the subsets

        decision_trees = []
        for i in range(self.n_estimators):
            tree = DecisionTree()
            tree.fit(subsets_x[i], subsets_y[i])
            decision_trees.append(tree)

        for i in range(self.n_estimators):
            decision_trees[0].print_tree()

        # decision_trees[0].print_tree()

    def _create_random_subsets(self, x, y):
        subsets_x = []
        subsets_y = []

        for i in range(self.n_estimators):
            subset_x = []
            subset_y = []
            index_shuf = list(range(round(len(x) * 0.66)))
            shuffle(index_shuf)

            for j in index_shuf:
                subset_x.append(x[j])
                subset_y.append(y[j])
            subsets_x.append(subset_x)
            subsets_y.append(subset_y)
        return np.array(subsets_x), np.array(subsets_y)

    def predict(self, x):

        # Predict the value in all the trees and it belongs to the most frequent result.

        raise NotImplementedError('Not implemented')

    def predict_proba(self, x):
        raise NotImplementedError('Not implemented')


forest = RandomForest(4)

iris = datasets.load_iris()
forest.fit(iris.data, iris.target)
# forest.fit([[3, 4], [2, 6], [4, 6], [1, 6], [9, 11], [21, 15]], ["T", "F", "T", "F", "T", "T"])
