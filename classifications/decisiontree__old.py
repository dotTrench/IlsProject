import numpy as np
from collections import Counter
from .classification_algorithm import ClassificationAlgorithm
from collections import Counter
import math


def check_is_unique(lst):
    return lst.all(lst[0])


class DecisionTree(ClassificationAlgorithm):
    def __init__(self):
        self.criterion = None
        self.max_featues = None
        self.max_depth = None
        self.samples_leaf = None
        self.laplace = None
        self._root = {}

    def _entropy(self, x, y):
        entropy = 0

        # Calc amount of results
        counts = Counter(y)

        # If all members belongs to the same class return entropy 0
        if len(counts) == 1:
            return entropy

        # calc entropy
        for key, val in counts.items():
            p = val / len(x)
            entropy = entropy - p * math.log2(p)

        return entropy

    def _gain(self, x, y, i):
        # TODO: Verify that this actually works
        # And make me less ugly

        val_freq = {}
        subset_entropy = 0.0

        for record in x:
            if record[i] in val_freq:
                val_freq[record[i]] += 1
            else:
                val_freq[record[i]] = 1
        # Calculate the sum of the entropy for each subset of records weighted
        # by their probability of occuring in the training set.
        for val in val_freq.keys():
            val_prob = val_freq[val] / sum(val_freq.values())
            data_subset = [record for record in x if record[i] == val]
            subset_entropy += val_prob * self._entropy(data_subset, y)

        # Subtract the entropy of the chosen attribute from the entropy of the
        # whole x set with respect to the target attribute (and return it)
        return (self._entropy(x, y) - subset_entropy)

    def _majority_value(self, y):
        # TODO: Actually implement me
        c = Counter(y)
        return y[0]

    def fit(self, x, y):
        self._root = self._build_tree(x, y)

    """ builds the actual tree recursivly """
    def _build_tree(self, x, y):
        if len(x[0]) <= 0:
            val = self._majority_value(y)
            return val

        if np.count_nonzero(y == y[0]) == len(y):
            # All the elements are the same return this classification
            return y[0]

        # choose_best_attribute(x, y)
        # best_attribute = _choose_best_attribute(x, y)
        # Choose best attribute to split on
        best = self._choose_best_attribute(x, y)

        tree = {best: {}}
        attribute_values = self.get_values(x, best)
        # Create new tree for each value in each column
        for val in attribute_values:
            new_x, new_y = self.get_examples(x, y, best, val)
            sub_tree = self._build_tree(new_x, new_y)
            tree[best][val] = sub_tree

        return tree

    """ Returns an integer value where the the split should occur """
    def _choose_best_attribute(self, x, y):
        # This is where the magic happens
        # TODO: Magic!
        cur_max = 0
        index = 0
        for i in range(len(x[0])):
            gain = self._gain(x, y, i)
            if gain > cur_max:
                cur_max = gain
                index = i

        return index

    def get_examples(self, x, y, attribute, val):
        # TODO: Please make me pretty
        new_x = []
        new_y = []
        for i in range(len(x)):
            entry = x[i]
            if entry[attribute] == val:
                new_entry = []
                new_y.append(y[i])
                for i in range(len(entry)):
                    if i != attribute:
                        new_entry.append(entry[i])
                new_x.append(new_entry)
        return np.array(new_x), np.array(new_y)

    def get_values(self, x, attribute):
        # TODO: Make me private
        return x[:, attribute]

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass
