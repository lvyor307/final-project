import pandas as pd


class FeaturesEngineering:
    def __init__(self, methods: list):
        self.methods = methods
        self.methods_dict = {'fourier_transform': self.fourier_transform}

    def fourier_transform(self, X: pd.DataFrame):
        pass

    def fit(self, X: pd.DataFrame):
        for method in self.methods:
            if method in self.methods_dict:
                X[method] = method(X)
