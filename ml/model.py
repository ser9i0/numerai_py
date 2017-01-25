import pandas as pd


class Model(object):
    """Abstract class for models. Every model should at least define:
        - identifier: ID of the model, as string (eg.: xgboost)
        - train: train the model and define self.model
        - predict: predict tournament data and save it to file"""
    def __init__(self, data, output_path):
        self.training_data = pd.read_csv(data['training_data'], float_precision='high')
        self.validation_data = pd.read_csv(data['validation_data'], float_precision='high')
        if 'reduced_features' in data:
            self.feature_cols = data['reduced_features']
        else:
            self.feature_cols = list(self.training_data.columns[:-1])
        self.tournament_file = data['tournament_file']
        self.output_path = output_path
        self.model = None

    def identifier(self):
        raise NotImplementedError("Identifier method not implemented yet.")

    def train(self):
        raise NotImplementedError("Train method not implemented yet.")

    def predict(self):
        raise NotImplementedError("Predict method not implemented yet.")
