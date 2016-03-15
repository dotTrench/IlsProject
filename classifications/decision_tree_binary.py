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
        t = DecisionTree()
        i = np.array([1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1])

        actual = t._majority_value(i)

        expected = 1
        self.assertEqual(actual, expected)


class TestAllValuesSame(unittest.TestCase):
    def test_returns_true_with_identical_values(self):
        t = DecisionTree()

        i = np.array([1, 1, 1])
        self.assertTrue(t._all_values_are_same(i))

    def test_returns_false_with_unidentical_values(self):
        t = DecisionTree()

        i = np.array([1, 2, 3])
        self.assertFalse(t._all_values_are_same(i))


class TestGini(unittest.TestCase):
    def test_gini(self):
        t = DecisionTree()

        i = np.array([1, 0, 0])
        actual = t._gini(i)
        expected = 0
        self.assertEqual(actual, expected)

        i = np.array([1, 1, 1])
        actual = t._gini(i)
        expected = 0.666666666

        self.assertAlmostEqual(actual, expected, places=7)


class GetSplitValuesTest(unittest.TestCase):
    def test_get_split_values(self):
        t = DecisionTree()

        i = np.array([5.1, 4.9, 6.4, 7.6])

        actual = t._get_split_values(i)

        expected = np.array([5.0, 5.75, 7.0])
        np.testing.assert_allclose(expected, actual)


class SplitValueGiniCalcTet(unittest.TestCase):
    def test_value_gini_calc(self):
        t = DecisionTree()

        col = np.array([5.1, 4.9, 6.4, 7.6])
        res = np.array([0, 0, 1, 2])
        actual = t._split_value_gini_calc(col, 5.0, res)

        self.assertAlmostEqual(actual, 0.4999995, places=4)


class BestSplitPointCalcTest(unittest.TestCase):
    def test_get_best_split_point(self):
        t = DecisionTree()
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        a, v = t._get_best_split_point(x, y)

        self.assertEqual(a, 0)
        self.assertEqual(v, 5.75)


class SplitTestCase(unittest.TestCase):
    def test_split(self):
        t = DecisionTree()

        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])
        ht_x, ht_y, lt_x, lt_y = t._split(x, y, 0, 5.75)

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


class Node:
    def __init__(self, feature=None, value=None):
        self.value = value
        self.feature = feature
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree():
    def __init__(self, criterion='gini', max_features=0, max_depth=0,
                 min_samples_leaf=0):

        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._root = None

    def fit(self, x, y):
        self._root = self._build_tree(x, y)

    def predict(self, x):
        return predict2(self._root, x)

    def _build_tree(self, x, y, depth=0):
        # If depth exceeds max_depth
        if depth > self.max_depth:
            return Node(value=self._majority_value(y))

        # If all the values in y are the same
        if self._all_values_are_same(y):
            return Node(value=y[0])

        # If there's no more attributes
        if len(x) <= 0:
            return Node(value=self._majority_value(y))

        split_feature, split_value = self._get_best_split_point(x, y)

        x1, y1, x2, y2 = self._split(x, y, split_feature, split_value)

        n = Node(feature=split_feature, value=split_value)
        n.left = self._build_tree(x1, y1, depth + 1)
        n.right = self._build_tree(x2, y2, depth + 1)

        return n

    def _split_value_gini_calc(self, column, value, results):
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
        return c.most_common(1)

    def _all_values_are_same(self, y):
        return len(np.unique(y)) == 1

    def _generate_features(self, row):
        return range(len(row))
        # return [i for i in range(0, len(row))]

    def _get_split_values(self, values):
        values = np.unique(values)

        return (values[:-1] + values[1:]) / 2

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

        ht_x = np.delete(ht_x, feature, 1)
        lt_x = np.delete(lt_x, feature, 1)

        return ht_x, ht_y, lt_x, lt_y

    def _predict2(self, node, x):
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
