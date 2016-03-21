from classifications.decisiontree import DecisionTree
import unittest
import numpy as np


class DecisionTreeTests(unittest.TestCase):
    def test_majority_value(self):
        t = DecisionTree()
        i = np.array([1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1])
        value, probability = t._majority_value(i)

        self.assertEqual(value, 1)
        self.assertEqual(probability, 0.5)

    def test_all_values_same_returns_true_with_identical_values(self):
        t = DecisionTree()

        i = np.array([1, 1, 1])
        self.assertTrue(t._all_values_are_same(i))

    def test_all_values_same_returns_false_with_unidentical_values(self):
        t = DecisionTree()

        i = np.array([1, 2, 3])
        self.assertFalse(t._all_values_are_same(i))

    def test_gini(self):
        t = DecisionTree()

        i = np.array([1, 0, 0])
        actual = t._gini(i)
        expected = 0
        self.assertEqual(actual, expected)

        i = np.array([1, 1, 1])
        actual = t._gini(i)
        expected = 0.666666666

        self.assertAlmostEqual(actual, 0.666666666)

    def test_get_split_values(self):
        t = DecisionTree()

        i = np.array([5.1, 4.9, 6.4, 7.6])

        actual = t._get_split_values(i)

        np.testing.assert_allclose(actual, [5.0, 5.75, 7.0])

    def test_value_gini_calc(self):
        t = DecisionTree()

        col = np.array([5.1, 4.9, 6.4, 7.6])
        res = np.array([0, 0, 1, 2])
        actual = t._split_value_gini_calc(col, 5.0, res)

        self.assertAlmostEqual(actual, 0.4999995, places=4)

    def test_get_best_split_point(self):
        t = DecisionTree()
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        a, v = t._get_best_split_point_efficent(x, y)

        self.assertEqual(a, 0)
        self.assertEqual(v, 5.75)

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

    def test_build_tree(self):
        # This test case just makes sure it doesn't crash, not really that
        # useful on its own
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        t = DecisionTree()
        t.fit(x, y)

    # This test case just makes sure it doesn't crash, not really that
    # useful on its own
    def test_predict(self):
        x = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.9, 4.3, 1.3],
            [7.6, 3.0, 6.6, 2.1]])
        y = np.array([0, 0, 1, 2])

        t = DecisionTree()
        t.fit(x, y)
        t.predict([[4.9, 3.0, 1.4, 0.2]])

if __name__ == '__main__':
    unittest.main()
