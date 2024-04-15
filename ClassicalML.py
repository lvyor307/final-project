import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import Utils


class Preprocessor:
    def __init__(self, methods: list):
        """
        :param methods: list of methods to apply to the audio files
        """
        self.methods = methods

    def run(self, list_of_audio_files: list, y: pd.DataFrame, index_name: str = 'filename') -> pd.DataFrame:
        """
        this method applies the methods to the audio files
        :param list_of_audio_files: list of audio files
        :param y: the target file
        :param index_name: the name of the index
        :return: the result of the methods
        """
        X = Utils.apply_methods(target_file=y, audio_files_list=list_of_audio_files, methods=self.methods)
        return X.set_index(index_name)


class BaseModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.best_params = None

    def train(self, X, y):
        if self.best_params:
            self.model.set_params(**self.best_params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def tune_hyperparameters(self, X, y, param_grid, cv=3, scoring='accuracy'):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        print(f"Best parameters for {self.name}: {self.best_params}")


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__('Random Forest', RandomForestClassifier(random_state=42))
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        self.param_grid = {
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2]
        }


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__('LightGBM', LGBMClassifier(random_state=42))
        self.param_grid = {
            'num_leaves': [10, 20],
            'min_data_in_leaf': [20, 50],
            'max_depth': [10, 20],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300]
        }


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('Logistic Regression',
                         LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42))
        self.param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2']  # 'l2' is the only penalty supported by the 'lbfgs' solver
        }


class ModelComparer:
    def __init__(self, models: list):
        self.models = models

    def tune_and_evaluate(self, X_train, y_train, X_test, y_test):
        results = {}
        for model in self.models:
            # Tune hyperparameters
            model.tune_hyperparameters(X_train, y_train, model.param_grid)
            # Train the model with the best parameters found
            model.train(X_train, y_train)
            # Evaluate the model on the test set
            accuracy = model.evaluate(X_test, y_test)
            results[model.name] = accuracy
        return results
