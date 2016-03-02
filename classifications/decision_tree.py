from .classification_algorithm import ClassificationAlgorithm
from collections import Counter
import math

def check_is_unique(lst):
    return lst.all(lst[0])


class Node:
    def __init__(self, value=None):
        self.value = value


class DecisionTree(ClassificationAlgorithm):
    def __init__(self):
        self.criterion = None
        self.max_featues = None
        self.max_depth = None
        self.samples_leaf = None
        self.laplace = None

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

    def _gain(self, x, y):



        entropy_s = self._entropy(x, y)
        entropy_t = 0
        pass

    def fit(self, x, y):
        # create node
        n = Node()

        # If all examples are positive, Return the single-node tree Root, with
        # label = +.
        # If all examples are negative, Return the single-node tree Root, with
        # label = -.
        if check_is_unique(y):
            print(y)
            # Whatever the common value is
            return Node(0)
        # If number of predicting attributes is empty, then Return the single
        # node tree Root,

        # with label = most common value of the target attribute in the
        # examples.
        for i in y:
            print(i)

        if len(x) <= 0:
            # Find the most common value in Y
            return Node(1)
        print('done')

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass
