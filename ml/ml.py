import pandas as pd
from sklearn.metrics import log_loss


class ML:
    def __init__(self, data):
        self.training_data = pd.read_csv(data['training_data'], float_precision='high')
        self.validation_data = pd.read_csv(data['validation_data'], float_precision='high')
        if 'reduced_features' in data:
            self.feature_cols = data['reduced_features']
        else:
            self.feature_cols = list(self.training_data.columns[:-1])
        self.tournament_data = pd.read_csv(data['tournament_file'], usecols=self.feature_cols, float_precision='high')

    def logistic_regression(self):
        """Create and fit logistic regression model"""
        from sklearn.linear_model import LogisticRegression

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
        # lr_score = log_regr.score(df_features_valid, df_target_valid)
        # print ":::: Mean accuracy score: {0}".format(lr_score)
        valid_predictions_proba = log_regr.predict_proba(df_features_valid)
        loss = log_loss(df_target_valid, valid_predictions_proba)
        print ":::: Log loss: {0}".format(loss)

        """Get probabilistic predictions for tournament data"""
        predictions_proba = log_regr.predict_proba(self.tournament_data)

        return predictions_proba[:,1]

    def xgboost(self):
        """Setting a XGBoost model"""
        import xgboost as xgb

        print ":: XGBoost ({0}) ::::".format(xgb.__version__)

        """Select all columns except last column (target)"""
        target_col = self.training_data.columns[-1]

        df_features_train = self.training_data[self.feature_cols]
        df_target_train = self.training_data[target_col]
        df_features_valid = self.validation_data[self.feature_cols]
        df_target_valid = self.validation_data[target_col]

        xgtrain = xgb.DMatrix(df_features_train.values, df_target_train.values)
        xgtest = xgb.DMatrix(df_features_valid.values, df_target_valid.values)
        xgpred = xgb.DMatrix(self.tournament_data.values)
        evallist = [(xgtest, 'eval'), (xgtrain, 'train')]

        print ":::: Training model with default settings..."
        """Specify parameters via map"""
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        param['eval_metric'] = 'logloss'
        num_round = 2
        xgb_model = xgb.train(param, xgtrain, num_round, evallist)

        """Check the accuracy on the validation set"""
        valid_predictions_proba = xgb_model.predict(xgtest)
        loss = log_loss(df_target_valid, valid_predictions_proba)
        print ":::: Log loss: {0}".format(loss)

        """Get probabilistic predictions for tournament data"""
        predictions_proba = xgb_model.predict(xgpred)

        return predictions_proba
