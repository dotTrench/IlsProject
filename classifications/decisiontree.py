from collections import Counter
import numpy as np
import math
import time
import operator
import bisect


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
            ret += 'Leaf:{0}:{1:.3g}%'.format(self.value,
                                              self.probability * 100)
        else:
            ret += 'Feature:{0} Value:{1}\n'.format(self.feature, self.value)
            ret += '\t' * level + 'L: ' + self.left.__str__(level + 1) + '\n'
            ret += '\t' * level + 'R: ' + self.right.__str__(level + 1) + '\n'

        return ret


class DecisionTree:
    def __init__(self, criterion='gini', max_features=None, max_depth=None,
                 min_samples_leaf=1):

        self.criterion = criterion                  # done
        self.max_features = max_features            # done
        self.max_depth = max_depth                  # done
        self.min_samples_leaf = min_samples_leaf    # TODO: Implement me

        self._root = None

    def get_params(self, deep=True):
        return {
            'criterion': self.criterion,
            'max_features': self.max_features,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }

    def set_params(self, criterion='gini', max_features=None, max_depth=None, min_samples_leaf=1):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        return self

    def fit(self, x, y):
        self._root = self._build_tree(x, y)

    def predict_proba(self, x):
        probabilities = [self.find(i).probability for i in x]
        return probabilities

    def predict(self, x):
        if len(x.shape) > 1:
            results = [self.find(v).value for v in x]
        else:
            results = self.find(x).value

        return results

    def find(self, x):
        """ Finds a node with which classifies input x"""
        if self._root is not None:
            return self._find(self._root, x)

    def _get_majority_node(self, y):
        """ Returns a node with a value of the majority value in y"""
        value, probability = self._majority_value(y)
        return Node(value=value, probability=probability)

    def _all_rows_equal(self, x):
        for i in range(len(x)):
            if not (x[0] == x[i]).all():
                return False
        return True

    def _build_tree(self, x, y, depth=0):

        # If depth exceeds max_depth
        if self.max_depth is not None and depth >= self.max_depth:
            return self._get_majority_node(y)

        # If all the values in y are the same
        if self._all_values_are_same(y):
            return self._get_majority_node(y)

        num_features = len(x[0])

        # If there are no more features
        if num_features <= 0:
            return self._get_majority_node(y)

        # If all the rows in X are identical
        if self._all_rows_equal(x):
            return self._get_majority_node(y)

        if len(x) <= self.min_samples_leaf:
            return self._get_majority_node(y)

        split_feature, split_value = self._get_best_split_point_efficent(x, y)
        if split_feature is None:
            return self._get_majority_node(y)

        x1, y1, x2, y2 = self._split(x, y, split_feature, split_value)

        n = Node(feature=split_feature, value=split_value)
        n.left = self._build_tree(x1, y1, depth + 1)
        n.right = self._build_tree(x2, y2, depth + 1)
        return n

    def _split_value_gini_calc(self, column, value, results):
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

        if sum(subset) == 0:
            print("ZEROZERO BAD!")
            return 0

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

    def _get_split_values(self, values):
        values = np.unique(values)
        return (values[:-1] + values[1:]) / 2

    def _get_best_split_point_efficent(self, x, y):
        features = self._generate_features(x[0])
        if self.max_features is not None and len(features) > self.max_features:
            features = range(self.max_features)

        best_gini = np.inf
        best_feature = None
        best_value = None
        for f in features:
            column = x[:, f]

            s = time.time()

            # Sort the column and the result
            lol = zip(column, y)

            lol = np.array(sorted(lol, key=operator.itemgetter(0)))

            sorted_column = lol[:, 0]
            sorted_result = lol[:, 1]

            # Calc first split-value distribution
            c = dict(Counter(sorted_result))

            dist = {}
            for key, value in c.items():
                temp_dist = {"lt": 0, "ht": value}
                dist[key] = temp_dist

            # print(dist)

            for index in range(0, len(sorted_column)):

                # Calc the distribution for each split-value
                dist = self._calc_new_distribution(dist, index, sorted_column, sorted_result)

                # Calc the gini for each distribution
                gini = self._calc_gini_for_distribution(dist)

                # print(gini)

                if gini < best_gini:
                    best_gini, best_feature, best_value = gini, f, sorted_column[index]
                    # print(f)
                    # print(sorted_column[index])
            e = time.time()

        # Calculate the correct split-value before returning
        print("Best feature: " + str(best_feature) + "  " + "Best value: " + str(best_value))
        return best_feature, best_value

    def _calc_new_distribution(self, prev_dist, index, column, result):

        # Won't work for the first iteration
        print(prev_dist)

        prev_result = result[index]

        prev_dist[prev_result]['ht'] -= 1
        prev_dist[prev_result]['lt'] += 1

        print(prev_dist)

        return prev_dist

    def _calc_gini_for_distribution(self, dist):

        ht_values = []
        lt_values = []

        for key, value in dist.items():
            ht_values.append(value['ht'])
            lt_values.append(value['lt'])

        ht_gini = self._gini(ht_values)
        lt_gini = self._gini(lt_values)

        lt_probability = sum(lt_values) / (sum(lt_values) + sum(ht_values))
        ht_probability = sum(ht_values) / (sum(lt_values) + sum(ht_values))

        full_gini = lt_gini * lt_probability + ht_gini * ht_probability

        return full_gini

    """
    def _split_value_gini_calc_efficent(self, column, value, results, lt_freq={}):
        ht_freq = {}

        ht_column = []
        ht_result = []

        # print(column)
        # print(results)

        # print(column)

        lol = zip(column, results)

        lol = np.array(sorted(lol, key=operator.itemgetter(0)))

        # print(lol)

        col = lol[:, 0]
        res = lol[:, 1]

        # print(lol)
        # print(col)
        # print(res)

        index = self.find_gt(col, value)
        # print("wtf")
        # print(index)

        ht_column = col[index:]
        ht_result = res[index:]
        lt_result = res[:index]

        ht_c = Counter(ht_result)
        lt_c = Counter(lt_result)

        ht_values = list(ht_c.values())
        # print("values")
        # print(list(lt_c.values()))
        # print(sum(list(lt_c.values())))
        lt_freq = sum(list(lt_c.values()))
        lt_values = list(lt_c.values())
        ht_gini = self._gini(ht_values)
        lt_gini = self._gini(lt_values)

        lt_probability = sum(lt_values) / (sum(lt_values) + sum(ht_values))
        ht_probability = sum(ht_values) / (sum(lt_values) + sum(ht_values))

        full_gini = lt_gini * lt_probability + ht_gini * ht_probability
        return full_gini, lt_freq, np.array(ht_column), np.array(ht_result)

    def _split_value_gini_calc_efficent(self, column, value, results, lt_freq={}):
        ht_freq = {}

        ht_column = []
        ht_result = []

        for i in range(len(results)):
            # print("         " + str(i))
            print("             " + str(column[i]))
            result = results[i]
            val = column[i]
            if val > value:
                ht_column.append(val)
                ht_result.append(result)
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
        return full_gini, lt_freq, np.array(ht_column), np.array(ht_result)
    """

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

        return np.array(ht_x), np.array(ht_y), np.array(lt_x), np.array(lt_y)

    def _find(self, node, x):
        if node.is_leaf():
            return node

        f = node.feature
        val = x[f]
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

    def find_gt(self, a, x):
        i = bisect.bisect_right(a, x)
        if i != len(a):
            return i
        raise ValueError