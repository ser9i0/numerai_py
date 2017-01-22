import os.path
import pandas as pd

class DimensionReduction:
    """Dimensionality reduction of features
    """

    def __init__(self, training_data):
        self.training_data = pd.read_csv(training_data, float_precision='high')
        self.feature_cols = list(self.training_data.columns[:-1])
        self.target_col = self.training_data.columns[-1]

    def low_variance(self):
        from sklearn.feature_selection import VarianceThreshold

        print ":: Reducing dimension: Low variance feature selection..."
        selector = VarianceThreshold(threshold=0.014)
        selector.fit(self.training_data[self.feature_cols])

        # get the indices of the features that are being kept
        feature_indices = selector.get_support(indices=True)

        # remove low-variance columns from index
        feature_names = [self.feature_cols[idx]
                         for idx, _
                         in enumerate(self.feature_cols)
                         if idx
                         in feature_indices]

        print ":::: Features selected: " + ", ".join(feature_names)

        return feature_names
        # http://stackoverflow.com/questions/29298973/removing-features-with-low-variance-scikit-learn
