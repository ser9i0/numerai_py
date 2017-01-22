import datetime
import glob
import numpy as np
import os
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys
import urllib2
import zipfile


class DataManager:
    """Class with necessary methods for datafiles download, processing and managing
    """

    def __init__(self):
        cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.output_path = os.path.join(cwd, 'data')
        self.data = {}

    def update_training_data(self):
        """Update datasets from the numer.ai server, comparing the last modification dates
        Return value:
         True -- if the datasets have been updated
         False -- if existing datasets are already up to date
        """
        dataset_url = 'https://api.numer.ai/competitions/current/dataset'

        req = urllib2.Request(dataset_url, headers={'User-Agent': 'UserAgent'})
        try:
            remote_file = urllib2.urlopen(req)
        except urllib2.URLError as err:
            print "/!\ ERROR: Error downloading files. Connection reset by peer."
            raise err

        rf_total_size = int(remote_file.info().getheader('Content-Length').strip())
        rf_datetime = datetime.datetime(*remote_file.info().getdate('last-modified')[0:6])
        prefix = rf_datetime.strftime('%Y%m%d')
        self.output_path = os.path.join(self.output_path, prefix)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        print ":: Checking last available datasets..."

        local_file_path = glob.glob(os.path.join(self.output_path, '*_numerai_training_data.csv'))

        if len(local_file_path) > 0 and os.path.isfile(local_file_path[0]):
            t = os.path.getmtime(local_file_path[0])
            lf_datetime = datetime.datetime.fromtimestamp(t)

            if rf_datetime <= lf_datetime:
                print ":::: Datasets are already up to date ({0}).".format(prefix)
                return False

        dataset_path = os.path.join(self.output_path,
                                    '{0}_dataset.zip'.format(prefix))

        print ":::: Downloading datasets ({0} bytes)...".format(rf_total_size)

        with open(dataset_path, 'wb') as dataset_file:
            chunk = (rf_total_size / 50) + 1
            sys.stdout.write(":::: Progress")
            progress_count = 0
            while True:
                buffer_data = remote_file.read(chunk)
                if not buffer_data:
                    break
                progress_count += 1
                sys.stdout.write(".")
                if progress_count is 25:
                    sys.stdout.write("50%")
                sys.stdout.flush()
                dataset_file.write(buffer_data)
            sys.stdout.write("100%!\r\n")
            dataset_file.close()

        """Remove any csv file (folder should be empty)"""
        filelist = glob.glob(os.path.join(self.output_path, '*.csv'))
        for f in filelist:
            os.remove(f)

        with zipfile.ZipFile(dataset_path) as zf:
            for element in zf.infolist():
                filename = os.path.split(element.filename)[1]

                if '._' in filename or '_data.csv' not in filename:
                    continue

                target_name = prefix + '_' + filename
                print ":::: Extracting {0}".format(target_name)

                zf.extract(element, self.output_path)
                os.rename(os.path.join(self.output_path, element.filename),
                          os.path.join(self.output_path, target_name))
            zf.close()
            os.remove(dataset_path)
            print ":::: Datasets successfully updated!"
            return True

    def check_datasets(self):
        """Function for cheking whether expected datafiles actually exist
        Returns a dictionary with the following elements:
         training_file -- Training dataset without any preprocessing
         tournament_file -- Dataset with the tournament data to be predicted and uploaded
         training_data -- (if exists) Training dataset after Adversary Validation process
         validation_data -- (if exists) Dataset for training validation
        """
        training_file = glob.glob(os.path.join(self.output_path, '*_numerai_training_data.csv'))
        tournament_file = glob.glob(os.path.join(self.output_path, '*_numerai_tournament_data.csv'))
        training_data = glob.glob(os.path.join(self.output_path, 'training_data.csv'))
        validation_data = glob.glob(os.path.join(self.output_path, 'validation_data.csv'))

        if len(training_file) is 1:
            self.data['training_file'] = training_file[0]
        else:
            print "ERROR: Training data not found."
            return False

        if len(tournament_file) is 1:
            self.data['tournament_file'] = tournament_file[0]
        else:
            print "ERROR: Prediction data not found."
            return False

        if len(training_data) is 1:
            self.data['training_data'] = training_data[0]

        if len(validation_data) is 1:
            self.data['validation_data'] = validation_data[0]

        return True

    def adv_validation_sets(self):
        """Function for generating validation datasets using Adversary Validation technique.
        It takes, from the training data, the 20% most similar elements to test data, in order
        to create a validation dataset. The other 80% is keeped as training data.
        Both datasets are also stored in 'data' dictionary as training_data and validation_data.
        """
        print ":: Creating validation datasets..."
        """Read original datasets"""
        df_train = pd.read_csv(self.data['training_file'], float_precision='high')
        df_test = pd.read_csv(self.data['tournament_file'], float_precision='high')

        """Select all columns except last column (target)"""
        feature_cols = list(df_train.columns[:-1])
        target_col = df_train.columns[-1]
        """Add new column TEST to differenciate train from test data"""
        test_col = 'TEST'
        df_train[test_col] = 0
        df_test[test_col] = 1
        """Concat both dataframes into df_data"""
        df_data = pd.concat([df_train, df_test])
        df_data = df_data.reindex_axis(feature_cols + [test_col, target_col], axis='columns')

        x_subset = df_data[feature_cols]
        y_subset = df_data[test_col]

        seed = 2380961743
        random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        predictions = np.zeros(y_subset.shape)
        n_folds = 5
        kfold = StratifiedKFold(y_subset, n_folds=n_folds, shuffle=True, random_state=seed)
        for i, (train_i, test_i) in enumerate(kfold):
            print ":::: Fold {0}/{1}:".format((i + 1), n_folds)

            x_subset_train = x_subset.iloc[train_i]
            y_subset_train = y_subset.iloc[train_i]

            x_subset_test = x_subset.iloc[test_i]
            y_subset_test = y_subset.iloc[test_i]

            random_forest.fit(x_subset_train, y_subset_train)

            p = random_forest.predict_proba(x_subset_test)[:, 1]
            auc = roc_auc_score(y_subset_test, p)
            print ":::: - AUC: {:.2f}".format(auc)

            predictions[test_i] = p

        """Sort predictions by value"""
        i = predictions.argsort()

        # Estaria bien saber cuantos datos de tipo training se clasifican como tipo TEST con p>0.5
        # Hay que filtrar por p>0.5 y coger los que sean TEST=0
        trainingp50 = 0
        for n in range(0, len(predictions)):
            if predictions[n] > 0.5 and df_data[test_col].iloc[n] == 0:
                trainingp50 += 1
        print ":::: Total number of {0} training elements are classified as test elements with p>0.5.".format(
            trainingp50)

        """Sort data by prediction confidence"""
        df_sorted = df_data.iloc[i]

        """Select only training data"""
        df_train_sorted = df_sorted.loc[df_sorted[test_col] == 0]

        """Drop unnecessary columns"""
        df_train_sorted = df_train_sorted.drop(test_col, axis=1)

        """Verify training data"""
        assert (df_train_sorted[target_col].sum() == df_train[target_col].sum())

        """Grab first 80% rows as train and last 20% rows as validation (those closest to test)"""
        validation_size = int(len(df_train_sorted) * 0.2)
        df_train = df_train_sorted.iloc[:-validation_size]
        df_valid = df_train_sorted.iloc[-validation_size:]
        print ":::: Creating dataset with validation size: {0}".format(validation_size)

        training_file_path = os.path.join(self.output_path, 'training_data.csv')
        validation_file_path = os.path.join(self.output_path, 'validation_data.csv')
        df_train.to_csv(training_file_path, index=False, float_format='%.15f')
        df_valid.to_csv(validation_file_path, index=False, float_format='%.15f')
        self.data['training_data'] = training_file_path
        self.data['validation_data'] = validation_file_path
        print ":::: Done"

    def write_tournament_data(self, pred_prob, model):
        t_id = 't_id'
        # df_prediction = pd.read_csv(self.data['tournament_file'], usecols=feature_cols, float_precision='high')
        # df_tournament = df_prediction[[t_id]]
        df_tournament = pd.read_csv(self.data['tournament_file'], usecols=[t_id], float_precision='high')
        # df_prediction = df_prediction.drop(t_id, axis=1)
        # print ":: Calculating predictions for tournament data..."
        df_tournament['probability'] = pd.Series(pred_prob, index=df_tournament.index)
        print ":: Saving prediction probabilities for model {0} into file...".format(model)
        tournament_file_path = os.path.join(self.output_path, '{0}_tournament_data.csv'.format(model))
        df_tournament.to_csv(tournament_file_path, header=[t_id, 'probability'], index=False)
        print ":: Done."
