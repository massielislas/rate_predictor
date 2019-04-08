import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(precision=4, suppress=True, floatmode='maxprec')

# manually selected features to remove, based on tests
DELETE_FEATURES = ['Life Expectancy', 'Life expectancy at birth (years)',
                   '2016 Crude Birth Rate', 'Use of basic sanitation services % (rural) 2015',
                   'urban use of basic sanitation services (%)',
                   'Use of basic sanitation services % (urban) 2015',
                   'Use of basic drinking water services % (rural) 2015.1',
                   'Use of basic drinking water services % (rural) 2015',
                   'Use of basic drinking water services % (urban) 2015',
                   'Reported Maternal mortality ratio',
                   'adolescent proportion of total population (%)'
                   ]

REGION = "Region"


class FeatureSelector:
    """
    FOR ANYONE WHO'S LOOKING AT THIS: you can test different hyperparameters by changing the values in the __init__()
    function; just follow my comments!
    """
    data: pd.DataFrame

    def __init__(self):
        # whether to delete features in DELETE_FEATURES or not
        self.delete_features = True

        # what dataset to load
        self.dataset_file = "Dataset in tabs - All Features.csv"

        # instantiate a scaler; set to None for no scaling
        # https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
        # I used a MinMaxScaler; alternatives are RobustScaler and Normalizer
        self.scaler = MinMaxScaler()

        # instantiate an imputer; set to None for no imputing (but this messes up scikit-learn)
        # scikit-learn documentation recommends imputing missing values (see link)
        # https://scikit-learn.org/stable/modules/impute.html#impute
        # I used a SimpleImputer with a "median" imputation strategy:
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
        # Alternatives are using a "mean", "most_frequent" imputation strategy, or "constant" strategy by supplying a
        #   fill value such as -1
        self.imputer = SimpleImputer(strategy="median")
        # self.imputer = SimpleImputer(strategy="constant", fill_value=-1)

        # whether to ignore the "Region" column (only the imputedData.csv file has it)
        self.ignore_region = True

        # Create feature selection objects; each selection object is run and the selected features are tallied
        # Alternatives are SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSet
        self.selection_methods = [
            SelectKBest(score_func=f_regression),
            SelectKBest(score_func=mutual_info_regression)
        ]

        self.X = None
        self.Y = None

    def main(self):
        self.load_dataset()
        self.scale()
        self.impute()
        self.select()

    def load_dataset(self):
        print("loading dataset:", self.dataset_file, "\n")
        self.data = pd.read_csv(self.dataset_file, index_col=0)
        if self.delete_features:
            self.data.drop(columns=DELETE_FEATURES, axis=1, inplace=True)
        # delete the "Region"  column from the data (only in the imputedData.csv dataset)
        if self.ignore_region and REGION in self.data.columns.values:
            del self.data[REGION]
        self.X = self.data[self.data.columns[1:]]
        self.Y = self.data[self.data.columns[0]]

    def scale(self):
        if self.scaler:
            self.X = self.scaler.fit_transform(self.X)

    def impute(self):
        if self.imputer:
            self.X = self.imputer.fit_transform(self.X)

    def select(self):
        selected = {}
        for selector in self.selection_methods:
            selector.fit(self.X, self.Y)
            selected_features = selector.get_support()
            # tally the selected features; first get a list of the feature names
            for feature in [self.data.columns[1:].values[i] for i in range(len(selected_features))
                            if selected_features[i]]:
                if feature in selected:
                    selected[feature] = selected[feature] + 1
                else:
                    selected[feature] = 1
        sorted_selection = sorted(selected.items(), key=lambda kv: kv[1])
        print(*sorted_selection, sep="\n")


if __name__ == '__main__':
    FeatureSelector().main()
