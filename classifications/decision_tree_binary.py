import numpy as np
from collections import Counter
import math
import time

# |4| # F
# |2| # T
# |4| # T


class Node:
    def __init__(self, value=None):
        self.value = value
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class DecisionTreeBinary:

    def __init__(self):
        pass

    def fit(self, x, y):

        self.root = self._build_tree(x, y, 0)

    def _build_tree(self, x, y, deep):
        print('BUILD TREE')
        print(len(x))
        if np.count_nonzero(y == y[deep]) == len(y):
            print('basfall')
            # All the elements are the same return this classification
            return Node(y[0])
        if deep >= len(x[0]):
            print('basfall')
            # TODO: Return majority value
            return Node(y[deep])

        print("ej basfall")
        # Get split-value for the first attribute in the x-list
        print('Deep: ' + str(deep))
        column = self.get_column(x, deep)
        index = self.calc_attribute_gini(column, y)
        print('Val: ' + str(column[index]))
        print('Ans: ' + str(y[index]))

        node = Node(column[index])

        node.left_child = Node(y[index])
        node.right_child = self._build_tree(x, y, deep+1)

        return node

    def get_column(self, data, column):
        return data[:, column]

    def gini(self, results):
        val = 0
        for res in results:
            val += (res / sum(results))**2
        return 1 - val

    # returns the split-value
    def calc_attribute_gini(self, attribute, results):
        max_gini = -float("inf")
        index = 0

        for i in range(0, len(attribute)):
            gini = self.calc_node_gini(attribute, results, i)
            print('Tot. gini: ' + str(gini))
            if gini > max_gini:
                max_gini = gini
                index = i

        return index

    def calc_node_gini(self, attribute, results, i):
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

        c1_gini = self.gini(child1_list)
        c2_gini = self.gini(child2_list)
        print('C1 GINI: {0}'.format(c1_gini))
        print('C2 GINI: {0}'.format(c2_gini))

        prob_1 = sum(child1_list) / (sum(child1_list) + sum(child2_list))
        prob_2 = sum(child2_list) / (sum(child1_list) + sum(child2_list))

        return c1_gini * prob_1 + c2_gini * prob_2

    def print_tree(self, node):

        print(str(node.value))

        if node.is_leaf():
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