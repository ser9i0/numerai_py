from model import Model
import os
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb


class XGB(Model):
    """XGBoost model"""
    def __init__(self, data, output_path):
        super(XGB, self).__init__(data, output_path)
        self.identifier = "xgboost"

    def identifier(self):
        return self.identifier

    def train(self):
        """Setting a XGBoost model"""
        print ":: XGBoost ({0}) ::::".format(xgb.__version__)

        """Select all columns except last column (target)"""
        target_col = self.training_data.columns[-1]

        df_features_train = self.training_data[self.feature_cols]
        df_target_train = self.training_data[target_col]
        df_features_valid = self.validation_data[self.feature_cols]
        df_target_valid = self.validation_data[target_col]

        xgtrain = xgb.DMatrix(df_features_train.values, df_target_train.values)
        xgtest = xgb.DMatrix(df_features_valid.values, df_target_valid.values)
        evallist = [(xgtest, 'eval'), (xgtrain, 'train')]

        print ":::: Training model with default settings..."
        """Specify parameters via map"""
        param = dict([('max_depth', 2), ('eta', 1), ('silent', 1), ('objective', 'binary:logistic')])
        param['eval_metric'] = 'logloss'
        num_round = 2
        self.model = xgb.train(param, xgtrain, num_round, evallist)

        """Check the accuracy on the validation set"""
        valid_predictions_proba = self.model.predict(xgtest)
        loss = log_loss(df_target_valid, valid_predictions_proba)
        print ":::: Log loss: {0}".format(loss)

    def predict(self):
        """Get probabilistic predictions for tournament data"""
        df_tournament = pd.read_csv(self.tournament_file, float_precision='high')
        print ":: Calculating predictions for tournament data..."
        xgpred = xgb.DMatrix(df_tournament.loc[:, self.feature_cols].values)
        predictions_proba = self.model.predict(xgpred)

        t_id = 't_id'
        df_prediction = pd.DataFrame(df_tournament.loc[:, t_id])
        df_prediction['probability'] = pd.Series(predictions_proba, index=df_prediction.index)
        print ":: Saving prediction probabilities into file..."
        tournament_file_path = os.path.join(self.output_path, '{0}_tournament_data.csv'.format(self.identifier))
        df_prediction.to_csv(tournament_file_path, header=[t_id, 'probability'], index=False)
        print ":: Done."

        return 1
