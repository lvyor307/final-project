import pandas as pd


class FeaturesEngineering:
    def __init__(self, methods: list):
        self.methods = methods
        self.methods_dict = {}

    def fit(self, X: pd.DataFrame):
        for method in self.methods:
            if method in self.methods_dict:
                X[method] = method(X)
