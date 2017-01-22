import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%matplotlib inline


# Anadir un fichero de estadisticas/analisis con las varianzas, correlaciones, etc.???

class DV:
    def __init__(self, training_file):
        self.training_file = pd.read_csv(training_file)
        self.feature_cols = list(self.training_file.columns[:-1])
        self.target_col = self.training_file.columns[-1]

    def histograms(self):
        print ":: Plotting histograms..."

        self.training_file.hist(column=self.feature_cols, bins=10, layout=(10, 5),
                                sharex=True, sharey=True)
        fig = plt.gcf()
        fig.set_size_inches(15, 18)
        fig.suptitle('Histograms', fontsize=22)
        fig.savefig('Histograms.png', dpi=192)
        plt.close()

    def variance(self):
        print ":: Variances:"
        df_variance = self.training_file.var()
        df_variance.sort_values(inplace=True)
        print df_variance

    def correlation(self, threshold=0.7):
        print ":: Plotting correlation heatmap..."

        correlations = self.training_file.corr(method='pearson')

        for row in correlations.itertuples():
            index = row[0]
            for col in correlations.loc[:, index:].columns.values:
                if col == index:
                    continue
                corr_value = getattr(row, col)
                if corr_value > threshold:
                    print ":::: {0} and {1} are highly correlated (>{2}) with r = {3}".format(index, col, threshold, corr_value)

        sns.heatmap(correlations, square=True)
        fig = plt.gcf()
        fig.set_size_inches(18, 16)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        fig.suptitle('Correlation Heatmap', fontsize=36)
        fig.savefig('CorrHeatmap.png', dpi=192)
        plt.close()
