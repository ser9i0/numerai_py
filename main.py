from sys import exit

from data_manager import DataManager
from ml.ml import ML

if __name__ == '__main__':

    data_manager = DataManager()
    is_updated = data_manager.update_training_data()

    check = data_manager.check_datasets()

    if check is False:
        exit(0)

    """If files have been updated, then generate new training and validation datasets"""
    if is_updated:
        data_manager.adv_validation_sets()

    ml = ML(data_manager.data)
    model = ml.logistic_regression()

    data_manager.write_tournament_data(model)
