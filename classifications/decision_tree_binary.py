import numpy as np
from collections import Counter
import math
import time

# |4| # F
# |2| # T
# |4| # T


class Node:
    def __init__(self, value=None, feature=None):
        self.value = value
        self.feature = feature
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class DecisionTreeBinary:

    def __init__(self):
        pass

    def fit(self, x, y):

        self.root = self._build_tree(x, y)

    def _build_tree(self, x, y):
        print('BUILD TREE')
        if np.count_nonzero(y == y[0]) == len(y):
            print('basfall')
            # All the elements are the same return this classification
            return Node(y[0])
        if len(x[0]) <= 0:
            print('basfall')
            # TODO: Return majority value
            return Node(y[0])

        print("ej basfall")
        # Find the best split
        best_feature, best_value = self.find_best_split(x, y)

        # Remove the attribute from the x-list

        x = np.delete(x, best_feature, 1)

        node = Node(best_value, best_feature)

        print(node.feature)
        print(node.value)

        node.left_child = Node(y[best_feature])
        node.right_child = self._build_tree(x, y)

        # node.left_child = Node(y[index])
        # node.right_child = self._build_tree(x, y, deep+1)

        #return node
        return node

    def gini(self, val1, val2):
        tot = val1 + val2
        return 1 - ((val1 / tot)**2 + (val2 / tot)**2)

    def calc_feature_gini(self, attributes, v, y):
        # calc gini for the feature compared to the mean-value
        val1 = 0
        val2 = 0

        for a in attributes:
            if a <= v:
                 val1 += 1
            else:
                 val2 += 1

        return self.gini(val1, val2)

    def find_best_split(self, x, y):
        n_instances = x.shape[0]
        n_features = x.shape[1]

        best_gini = np.inf
        best_feature, best_value = 0, 0

        for f in range(n_features):
            values = np.unique(x[:, f])
            values = (values[:-1] + values[1:]) / 2
            for v in values:
                gini = self.calc_feature_gini(x[f], v, y)
                if gini < best_gini:
                    best_gini, best_feature, best_value = gini, f, v

        return best_feature, best_value

        print(str(node.value))

        if node.is_leaf():
            return
        else:
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)

    def print_tree(self, node):

        print(node.feature)
        print(node.value)

        if node.is_leaf():
            print()
            return
        else:
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)

    def predict(self, x):
        return self._predict2(x, self.root, 0)

    def _predict2(self, dataset, node, deep):

        print()
        print(dataset)
        print(node.value)
        print(deep)

        if node.is_leaf():
            print('leaf-node')
            return node.value
        elif dataset[deep] <= node.value:
            return self._predict2(dataset, node.left_child, deep+1)
        else:
            return self._predict2(dataset, node.right_child, deep+1)