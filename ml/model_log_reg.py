from model import Model
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class LogReg(Model):
    """Logistic Regression model (baseline model)"""
    def __init__(self, data, output_path):
        super(LogReg, self).__init__(data, output_path)
        self.identifier = "log_reg"

    def identifier(self):
        return self.identifier

    def train(self):
        """Create and fit logistic regression model"""
        print ":: Baseline Model - Logistic Regression ::::"

        """Select all columns except last column (target)"""
        target_col = self.training_data.columns[-1]

        df_features_train = self.training_data[self.feature_cols]
        df_target_train = self.training_data[target_col]
        df_features_valid = self.validation_data[self.feature_cols]
        df_target_valid = self.validation_data[target_col]

        print ":::: Training model with default settings..."
        self.model = LogisticRegression()
        self.model = self.model.fit(df_features_train, df_target_train)

        """Check the accuracy on the validation set"""
        # lr_score = log_regr.score(df_features_valid, df_target_valid)
        # print ":::: Mean accuracy score: {0}".format(lr_score)
        valid_predictions_proba = self.model.predict_proba(df_features_valid)
        loss = log_loss(df_target_valid, valid_predictions_proba)
        print ":::: Log loss: {0}".format(loss)

    def predict(self):
        """Get probabilistic predictions for tournament data"""
        df_tournament = pd.read_csv(self.tournament_file, float_precision='high')
        print ":: Calculating predictions for tournament data..."
        predictions_proba = self.model.predict_proba(df_tournament.loc[:, self.feature_cols])
        predictions_proba = predictions_proba[:, 1]

        t_id = 't_id'
        df_prediction = pd.DataFrame(df_tournament.loc[:, t_id])
        df_prediction.loc[:, 'probability'] = pd.Series(predictions_proba, index=df_prediction.index)
        print ":: Saving prediction probabilities into file..."
        tournament_file_path = os.path.join(self.output_path, '{0}_tournament_data.csv'.format(self.identifier))
        df_prediction.to_csv(tournament_file_path, header=[t_id, 'probability'], index=False)
        print ":: Done."

        return 1
