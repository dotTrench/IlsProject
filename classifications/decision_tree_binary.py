import numpy as np
from collections import Counter
import math
import time


class Node:
    def __init__(self, value=None):
        self.value = value
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

# |4| # F
# |2| # T
# |4| # T


def get_best_attribute():
    return 0


def build_tree(dataset, results, depth=0):
    depth += 1
    print(dataset)
    print(results)
    if depth > 10:
        print('too deppp')
        return Node(0)
    if np.count_nonzero(results == results[0]) == len(results):
        print('results')
        # All the elements are the same return this classification
        return Node(results[0])
    if len(dataset) <= 0:
        print('XDDD')

        # TODO: Return majority value
        return Node(results[0])

    attributes = [i for i in range(0, len(dataset[0]))]
    best = get_best_attribute()

    column = get_column(dataset, best)
    index = get_split_index(column, results)
    value = column[index]

    new_left_dataset = []
    new_left_results = []
    for i in range(len(dataset)):
        entry = dataset[i]
        if entry[best] < value:
            new_entry = []
            new_left_results.append(results[i])
            for i in range(len(entry)):
                if i != best:
                    new_entry.append(entry[i])
            if new_entry != []:
                new_left_dataset.append(new_entry)

    node = Node(value)
    node.left_child = build_tree(np.array(new_left_dataset), np.array(new_left_results), depth)
    #node.right_child = build_tree(np.array(n_dataset), np.array(n_results), depth)

    # Remove column
    # node.left_child = build_tree(dataset, results)
    # node.right_child = build_tree(dataset, results)

    return node


def get_column(data, column):
    return data[:, column]


def g_calc(dataset, results):
    ginis = []
    column = None
    for i in range(len(dataset[0])):
        column = get_column(dataset, i)
        ginis.append(gini_calc(column, results))

    x = max(ginis)
    y = ginis.index(x)

    return ginis


def gini(results):
    val = 0
    for res in results:
        val += (res / sum(results))**2
    return 1 - val


def get_split_index(column, results):
    max_gini = -float("inf")

    index = 0
    for i in range(0, len(column)):
        gini = calc_one_gini(column, results, i)
        if gini > max_gini:
            max_gini = gini
            index = i

    return index


def calc_one_gini(attribute, results, i):
    a = attribute[i]
    child1 = {}
    child2 = {}
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

    child1_list = list(child1.values())
    child2_list = list(child2.values())

    c1_gini = gini(child1_list)
    c2_gini = gini(child2_list)

    prob_1 = sum(child1_list) / (sum(child1_list) + sum(child2_list))
    prob_2 = sum(child2_list) / (sum(child1_list) + sum(child2_list))

    return c1_gini * prob_1 + c2_gini * prob_2




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

