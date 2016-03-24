from classifications.decisiontree import DecisionTree
from random import shuffle, randint
from sklearn import datasets
from collections import Counter
import numpy as np


class RandomForest:
    def __init__(self, n_estimators=10, criterion='gini', max_features=None,
                 max_depth=None, min_samples_leaf=1, bagging=False,
                 sample_size=1.0):

        self.criterion = criterion                  # done
        self.max_features = max_features            # done
        self.max_depth = max_depth                  # done
        self.min_samples_leaf = min_samples_leaf    # done
        self.bagging = bagging                      # TODO: Please implement me
        self.sample_size = sample_size              # done
        self.n_estimators = n_estimators            # done

        self._decision_trees = []

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_features': self.max_features,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'bagging': self.bagging,
            'sample_size': self.sample_size
        }

    def fit(self, x, y):
        subsets_x, subsets_y = self._create_random_subsets(x, y)

        for i in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion,
                                max_features=self.max_features,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf)

            print(i, self.n_estimators)
            tree.fit(subsets_x[i], subsets_y[i])
            self._decision_trees.append(tree)

    def predict(self, x):
        result = []
        for i in x:
            results = [t.predict(i) for t in self._decision_trees]
            c = Counter(results)
            val, _ = c.most_common(1)[0]
            result.append(val)
        return result

    def predict_proba(self, x):
        nodes = [t.find(x) for t in self._decision_trees]

        values = [n.value for n in nodes]

        c = Counter(values)
        val, _ = c.most_common(1)[0]

        probabilities = [n.probability for n in nodes if n.value == val]

        return sum(probabilities) / len(values)

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


if __name__ == '__main__':
    forest = RandomForest(n_estimators=1000)
    tree = DecisionTree()

    iris = datasets.load_iris()
    # tree.fit(iris.data, iris.target)
    # tree.print_tree()
    forest.fit(iris.data, iris.target)
    r = forest.predict_proba(iris.data[19])
    print('{0:.5}%'.format(r * 100))
