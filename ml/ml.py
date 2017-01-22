import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class ML:
    def __init__(self, data):
        self.training_data = pd.read_csv(data['training_data'], float_precision='high')
        self.validation_data = pd.read_csv(data['validation_data'], float_precision='high')
        if 'reduced_features' in data:
            self.feature_cols = data['reduced_features']
        else:
            self.feature_cols = list(self.training_data.columns[:-1])

    def logistic_regression(self):
        """Create and fit logistic regression model"""
        print ":: Baseline Model - Logistic Regression ::::"
        """Select all columns except last column (target)"""

        target_col = self.training_data.columns[-1]

        df_features_train = self.training_data[self.feature_cols]
        df_target_train = self.training_data[target_col]
        df_features_valid = self.validation_data[self.feature_cols]
        df_target_valid = self.validation_data[target_col]

        print ":::: Training model with default settings..."
        log_regr = LogisticRegression()
        log_regr = log_regr.fit(df_features_train, df_target_train)

        """Check the accuracy on the validation set"""
        lr_score = log_regr.score(df_features_valid, df_target_valid)
        print ":::: Mean accuracy score: {0}".format(lr_score)
        predictions_proba = log_regr.predict_proba(df_features_valid)
        loss = log_loss(df_target_valid, predictions_proba)
        print ":::: Log loss: {0}".format(loss)

        return log_regr
