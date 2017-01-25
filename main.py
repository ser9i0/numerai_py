import numpy as np
from sys import exit

from data_manager import DataManager
from fe.data_visualization import DV
from fe.dimension_reduction import DimensionReduction
# Models
from ml.ml import ML
from ml.model_log_reg import LogReg
from ml.model_xgboost import XGB


if __name__ == '__main__':

    data_manager = DataManager()
    is_updated = data_manager.update_training_data()

    check = data_manager.check_datasets()

    if check is False:
        exit(0)

    """If files have been updated, then generate new training and validation datasets"""
    if is_updated:
        data_manager.adv_validation_sets()

    dr = DimensionReduction(data_manager.data['training_data'])
    data_manager.data['reduced_features'] = dr.low_variance()

    """Baseline model: Logistic Regression"""
    log_reg = LogReg(data_manager.data, data_manager.output_path)
    models = np.array([log_reg])

    """XGBoost"""
    xgb = XGB(data_manager.data, data_manager.output_path)
    models = np.append(models, [xgb])

    """Train models"""
    ml = ML(models)
    ml.run()

    # data_manager.write_tournament_data(lr_preds, 'logreg')
    # data_manager.write_tournament_data(xgb_preds,'xgboost')

    dv = DV(data_manager.data['training_file'])
    # dv.histograms()
    # dv.correlation()
    # dv.variance()
