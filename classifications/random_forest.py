from decisiontree import DecisionTree
from random import shuffle, randint
import numpy as np
from sklearn import datasets
from collections import Counter


class RandomForest:
    def __init__(self, n_estimators=10, criterion='gini', max_features=None,
                 max_depth=None, min_samples_leaf=1, bagging=False,
                 sample_size=1.0):

        self.criterion = criterion                  # Done
        self.max_features = max_features            # Done
        self.max_depth = max_depth                  # Done
        self.min_samples_leaf = min_samples_leaf    # Done
        self.bagging = bagging
        self.sample_size = sample_size              # Done
        self.n_estimators = n_estimators            # Done

        self._decision_trees = []

    def fit(self, x, y):

        # Create n random subsets from the x-list
        # Each tree is usualy trained on 2/3 of the data

        subsets_x, subsets_y = self._create_random_subsets(x, y)

        for i in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion,
                                max_features=self.max_features,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf)

            tree.fit(subsets_x[i], subsets_y[i])
            self._decision_trees.append(tree)

    def _generate_subset(self, num_samples, x, y):
        subset_x = []
        subset_y = []
        for i in range(num_samples):
            r = randint(0, len(x) - 1)
            subset_x.append(x[r])
            subset_y.append(y[r])

        return subset_x, subset_y

    def _create_random_subsets(self, x, y):
        subsets_x = []
        subsets_y = []

        num_samples = round(len(x) * self.sample_size)
        for i in range(self.n_estimators):
            sub_x, sub_y = self._generate_subset(num_samples, x, y)
            subsets_x.append(sub_x)
            subsets_y.append(sub_y)

        return np.array(subsets_x), np.array(subsets_y)

    def predict(self, x):
        results = [t.predict(x) for t in self._decision_trees]
        c = Counter(results)
        val, frequency = c.most_common(1)[0]

        return val

    def predict_proba(self, x):
        results = [t.predict_proba(x) for t in self._decision_trees]


if __name__ == '__main__':
    forest = RandomForest()
    tree = DecisionTree()

    iris = datasets.load_iris()
    # tree.fit(iris.data, iris.target)
    # tree.print_tree()
    forest.fit(iris.data, iris.target)
    r = forest.predict(iris.data[0])
    print(r)
