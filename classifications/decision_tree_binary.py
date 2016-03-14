import numpy as np
from collections import Counter
import math
import time
import unittest
import numpy as np
from sklearn import datasets


class TestPredict():
    def test_predict(self):
        t = DecisionTree()
        dataset = datasets.load_iris()
        t.fit(dataset.data, dataset.target)
        r = t.predict([4.9, 3.0, 1.4, 0.2])
        self.assertEqual(r, 0)


class TestMajorityValue():
    def test_majority_value(self):
        i = np.array([1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1])

        actual = majority_value(i)

        expected = 1
        self.assertEqual(actual, expected)


class TestAllValuesSame(unittest.TestCase):
    def test_returns_true_with_identical_values(self):
        i = np.array([1, 1, 1])
        self.assertTrue(all_values_are_same(i))

    def test_returns_false_with_unidentical_values(self):
        i = np.array([1, 2, 3])
        self.assertFalse(all_values_are_same(i))


class TestGini(unittest.TestCase):
    # Some real tests here
    def test_gini(self):
        i = np.array([1, 0, 0])
        actual = gini(i)
        expected = 0
        self.assertEqual(actual, expected)

        i = np.array([1, 1, 1])
        actual = gini(i)
        expected = 0.666666666

        self.assertAlmostEqual(actual, expected, places=7)


class GetSplitValuesTest(unittest.TestCase):
    def test_get_split_values(self):
        i = np.array([5.1, 4.9, 6.4, 7.6])
        i = np.unique(i)

        actual = get_split_values(i)

        expected = np.array([5.0, 5.75, 7.0])
        np.testing.assert_allclose(expected, actual)


class SplitValueGiniCalcTet(unittest.TestCase):
    def test_value_gini_calc(self):
        col = np.array([5.1, 4.9, 6.4, 7.6])
        res = np.array([0, 0, 1, 2])
        actual = split_value_gini_calc(col, 5.0, res)

        self.assertAlmostEqual(actual, 0.4999995, places=4)


class BestSplitPointCalcTest(unittest.TestCase):
    def test_get_best_split_point(self):
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        a, v = get_best_split_point(x, y)

        self.assertEqual(a, 0)
        self.assertEqual(v, 5.75)


class SplitTestCase(unittest.TestCase):
    def test_split(self):
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])
        ht_x, ht_y, lt_x, lt_y = split(x, y, 0, 5.75)

        ht_x_expected = np.array([
            [2.9, 4.3, 1.3],
            [3.0, 6.6, 2.1]
        ])
        ht_y_expected = np.array([1, 2])

        np.testing.assert_allclose(ht_x_expected, ht_x)
        np.testing.assert_allclose(ht_y_expected, ht_y)

        lt_x_expected = np.array([
            [3.5, 1.4, 0.2],
            [3.0, 1.4, 0.2]
        ])

        lt_y_expected = np.array([0, 0])

        np.testing.assert_allclose(lt_x_expected, lt_x)
        np.testing.assert_allclose(lt_y_expected, lt_y)


class BuildTreeTestCase(unittest.TestCase):
    def test_build_tree(self):
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        t = DecisionTree()
        t.fit(x, y)


def split_value_gini_calc(column, value, results):
    """ returns the gini value column is split at value("value") """
    from pprint import pprint
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

    ht_gini = gini(ht_values)
    lt_gini = gini(lt_values)

    lt_probability = sum(lt_values) / (sum(lt_values) + sum(ht_values))
    ht_probability = sum(ht_values) / (sum(lt_values) + sum(ht_values))

    full_gini = lt_gini * lt_probability + ht_gini * ht_probability
    return full_gini


def gini(subset):
    val = 0
    for res in subset:
        val += (res / sum(subset))**2
    return 1 - val


def majority_value(y):
    c = Counter(y)
    return c.most_common(1)


def all_values_are_same(y):
    return len(np.unique(y)) == 1


def generate_features(row):
    return [i for i in range(0, len(row))]


def get_split_values(values):
    values = np.unique(values)

    return (values[:-1] + values[1:]) / 2


def get_best_split_point(x, y):
    features = generate_features(x[0])

    best_gini = np.inf
    best_feature = None
    best_value = None

    for f in features:
        column = x[:, f]

        split_values = get_split_values(column)
        for v in split_values:
            gini = split_value_gini_calc(column, v, y)
            if gini < best_gini:
                best_gini, best_feature, best_value = gini, f, v

    return best_feature, best_value


def split(x, y, feature, value):
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

    ht_x = np.delete(ht_x, feature, 1)
    lt_x = np.delete(lt_x, feature, 1)

    return ht_x, ht_y, lt_x, lt_y


def build_tree(x, y, depth=0):
    all_same = all_values_are_same(y)
    if all_values_are_same(y):
        return Node(value=y[0])
    if len(x) <= 0:
        return Node(value=majority_value(y))

    split_feature, split_value = get_best_split_point(x, y)

    x1, y1, x2, y2 = split(x, y, split_feature, split_value)

    n = Node(feature=split_feature, value=split_value)
    n.left = build_tree(x1, y1)
    n.right = build_tree(x2, y2)

    return n

    # If all values in Y are the same, return Y
    # if len(x) >= 0: return majority_value(y)

    # j = attribute index?
    # s = split value
    # Select split value j,s
    # Split y on j, s
    # Build tree on left child where values are smaller than y
    # Build tree on right child where values are greater or equal to y


class Node:
    def __init__(self, feature=None, value=None):
        self.value = value
        self.feature = feature
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree():
    def __init__(self):
        self._root = None

    def fit(self, x, y):
        self._root = build_tree(x, y)

    def predict(self, x):
        return predict2(self._root, x)


def predict2(node, x):
    if node.is_leaf():
        return node.value

    f = node.feature
    val = x[f]
    x = np.delete(x, f, 0)

    if val <= node.value:
        return predict2(node.left, x)
    else:
        return predict2(node.right, x)

if __name__ == '__main__':
    # x = np.array([
    #         [5.1, 3.5, 1.4, 0.2],
    #         [4.9, 3.0, 1.4, 0.2],
    #         [6.4, 2.9, 4.3, 1.3],
    #         [7.6, 3.0, 6.6, 2.1]])
    # y = np.array([0, 0, 1, 2])

    # t = DecisionTree()
    # t.fit(x, y)
    # # print(t._root.value)
    # r = t.predict([6.4, 2.9, 4.3, 1.3])
    # print(r)
    unittest.main()
