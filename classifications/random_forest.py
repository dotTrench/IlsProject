from .classification_algorithm import ClassificationAlgorithm


class RandomForest(ClassificationAlgorithm):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.criterion = None
        self.max_featues = None
        self.max_depth = None
        self.max_samples_leaf = None
        self.laplace = None
        self.n_estimators = None
        self.bagging = None
        self.sample_size = None

    def fit(self, x, y):
        raise NotImplementedError('Not implemented')

    def predict(self, x):
        raise NotImplementedError('Not implemented')

    def predict_proba(self, x):
        raise NotImplementedError('Not implemented')
