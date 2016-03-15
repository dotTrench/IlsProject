import numpy as np
from collections import Counter
import math
import time
import unittest
import numpy as np
from sklearn import datasets


class Node:
    def __init__(self, feature=None, value=None, probability=None):
        self.value = value
        self.feature = feature
        self.probability = probability

        # Children
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self, level=0):
        ret = '\t' * level
        if self.is_leaf():
            ret += 'Leaf:{0}:{1}%'.format(self.value, (self.probability * 100))
        else:
            ret += 'Feature:{0} Value:{1}\n'.format(self.feature, self.value)
            ret += '\t' * level + 'L: ' + self.left.__str__(level + 1) + '\n'
            ret += '\t' * level + 'R: ' + self.right.__str__(level + 1) + '\n'

        return ret


class DecisionTree:
    def __init__(self, criterion='gini', max_features=None, max_depth=None,
                 min_samples_leaf=1):

        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._root = None

    def fit(self, x, y):
        self._root = self._build_tree(x, y)

    def predict_proba(self, x):
        return self._find(self._root, x).probability

    def predict(self, x):
        return self._find(self._root, x).value

    def _get_majority_node(self, y):
        """ Returns a node with a value of the majority value in y"""
        value, probability = self._majority_value(y)
        return Node(value=value, probability=probability)

    def _build_tree(self, x, y, depth=0):
        # If depth exceeds max_depth
        if self.max_depth is not None and depth > self.max_depth:
            return self._get_majority_node(y)

        # If all the values in y are the same
        if self._all_values_are_same(y):
            return Node(value=y[0], probability=1)

        # Not sure if this is how it's meant to be done
        if self._all_values_are_same(x):
            return self._get_majority_node(y)

        num_features = len(x[0])

        # If num_features the max features set in the constructor
        if self.max_features is not None and num_features > self.max_features:
            return self._get_majority_node(y)

        # If there's no more features
        if num_features <= 0:
            return self._get_majority_node(y)

        # Else perform a split
        else:
            split_feature, split_value = self._get_best_split_point(x, y)
            x1, y1, x2, y2 = self._split(x, y, split_feature, split_value)

            n = Node(feature=split_feature, value=split_value)
            n.left = self._build_tree(x1, y1, depth + 1)
            n.right = self._build_tree(x2, y2, depth + 1)

            return n

    def _split_value_gini_calc(self, column, value, results):
        """ returns the gini value column is split at value("value") """
        lt_freq = {}
        ht_freq = {}

        for i in range(len(results)):
            result = results[i]
            val = column[i]
            if val > value:
                if result in ht_freq:
                    ht_freq[result] += 1
                else:
                    ht_freq[result] = 1
            else:
                if result in lt_freq:
                    lt_freq[result] += 1
                else:
                    lt_freq[result] = 1

        ht_values = list(ht_freq.values())
        lt_values = list(lt_freq.values())

        ht_gini = self._gini(ht_values)
        lt_gini = self._gini(lt_values)

        lt_probability = sum(lt_values) / (sum(lt_values) + sum(ht_values))
        ht_probability = sum(ht_values) / (sum(lt_values) + sum(ht_values))

        full_gini = lt_gini * lt_probability + ht_gini * ht_probability
        return full_gini

    def _gini(self, subset):
        val = 0
        for res in subset:
            val += (res / sum(subset))**2
        return 1 - val

    def _majority_value(self, y):
        c = Counter(y)
        most_common = c.most_common(1)
        value, amount = most_common[0]

        elements = list(c.elements())
        probability = amount / len(elements)

        return value, probability

    def _all_values_are_same(self, y):
        return len(np.unique(y)) == 1

    def _generate_features(self, row):
        return range(len(row))
        # return [i for i in range(0, len(row))]

    def _get_split_values(self, values):
        values = np.unique(values)
        return (values[:-1] + values[1:]) / 2
#        return np.array(values[0])

    def _get_best_split_point(self, x, y):
        features = self._generate_features(x[0])
        best_gini = np.inf
        best_feature = None
        best_value = None

        for f in features:
            column = x[:, f]
            split_values = self._get_split_values(column)
            for v in split_values:
                gini = self._split_value_gini_calc(column, v, y)
                if gini < best_gini:
                    best_gini, best_feature, best_value = gini, f, v

        return best_feature, best_value

    def _split(self, x, y, feature, value):
        col = x[:, feature]
        ht_x = []
        ht_y = []

        lt_x = []
        lt_y = []

        for i in range(len(col)):
            val = col[i]
            if val > value:
                ht_x.append(x[i])
                ht_y.append(y[i])
            else:
                lt_x.append(x[i])
                lt_y.append(y[i])

        if len(ht_x) > 0:
            ht_x = np.delete(ht_x, feature, 1)
        if len(lt_x) > 0:
            lt_x = np.delete(lt_x, feature, 1)

        return np.array(ht_x), np.array(ht_y), np.array(lt_x), np.array(lt_y)

    def _find(self, node, x):
        if node.is_leaf():
            return node

        f = node.feature
        val = x[f]
        x = np.delete(x, f, 0)

        if val <= node.value:
            return self._find(node.left, x)
        else:
            return self._find(node.right, x)

    def print_tree(self):
        if self._root is not None:
            print(self._root)

    def __str__(self):
        if self._root is not None:
            return self._root.__str__()

        return self

if __name__ == '__main__':
    unittest.main()