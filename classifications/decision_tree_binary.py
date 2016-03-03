import numpy as np
from collections import Counter
import math
import time


# |4| # F
# |2| # T
# |4| # T

def get_column(data, column):
    return data[:, column]


def g_calc(dataset, results):
    ginis = []
    column = None
    for i in range(len(dataset[0])):
        column = get_column(dataset, i)
        print(column)
        ginis.append(gini_calc(column, results))

    x = max(ginis)
    print(x)
    y = ginis.index(x)
    print(y)
    print(column[1])

    # print(ginis)
    return ginis


def gini(results):
    val = 0
    for res in results:
        val += (res / sum(results))**2
    return 1 - val


def get_max_gini_index(dataset, results):
    max_gini = -float("inf")

    index = 0
    for i in range(0, len(dataset[0])):
        column = get_column(dataset, i)
        print('col :' + column)
        gini = calc_one_gini(column, results, i)
        if gini > max_gini:
            max_gini = gini
            index = i

    return index

def calc_one_gini(attribute, results, i):
    a = attribute[i]
    child1 = {}
    child2 = {}
    print('A: ' + str(a))
    for j in range(0, len(attribute)):
        b = attribute[j]
        if b <= a:
            if results[j] in child1:
                child1[results[j]] += 1
            else:
                child1[results[j]] = 1
        else:
            if results[j] in child2:
                child2[results[j]] += 1
            else:
                child2[results[j]] = 1

    print('Child 1: {0}'.format(child1))
    print('Child 2: {0}'.format(child2))
    print()

    child1_list = list(child1.values())
    child2_list = list(child2.values())

    c1_gini = gini(child1_list)
    c2_gini = gini(child2_list)
    print('C1 GINI: {0}'.format(c1_gini))
    print('C2 GINI: {0}'.format(c2_gini))

    prob_1 = sum(child1_list) / (sum(child1_list) + sum(child2_list))
    prob_2 = sum(child2_list) / (sum(child1_list) + sum(child2_list))

    return c1_gini * prob_1 + c2_gini * prob_2


class Node:
    def __init__(self, value=None):
        self.value = value
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class DecisionTreeBinary:
    def fit(self, x, y):
        for i in range(len(x)):
            print('{0} = {1}'.format(x[i], y[i]))

        self._build_tree(x, y)
        pass

    def _get_column(self, data, column):
        return data[:, column]

    def _build_tree(self, x, y):
        if np.count_nonzero(y == y[0]) == len(y):
            # All the elements are the same return this classification
            return Node(y[0])
        if len(x) <= 0:
            # TODO: Return majority value
            return Node(y[0])

        attributes = [i for i in range(0, len(x[0]))]
        best = self._choose_best_attribute(x, y)
        values = self._get_column(x, best)
        possible_results = np.unique(y)

