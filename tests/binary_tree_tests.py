from .classifications import *
import unittest


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

if __name__ == '__main__':
    unittest.main()
