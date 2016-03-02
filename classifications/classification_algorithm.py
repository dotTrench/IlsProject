from abc import ABCMeta, abstractmethod


class ClassificationAlgorithm(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass
