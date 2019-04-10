import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

import compare_lists

np.set_printoptions(precision=4, suppress=True, floatmode='maxprec')

# manually selected features to remove, based on tests
DELETE_FEATURES = ['Life Expectancy',
                   'Life expectancy at birth (years)',
                   # '2016 Crude Birth Rate',
                   'Use of basic sanitation services % (rural) 2015',
                   # 'urban use of basic sanitation services (%)',
                   'Use of basic sanitation services % (urban) 2015',
                   'Use of basic drinking water services % (rural) 2015.1',
                   'Use of basic drinking water services % (rural) 2015',
                   'Use of basic drinking water services % (urban) 2015',
                   'Reported Maternal mortality ratio',
                   # 'adolescent proportion of total population (%)'
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
        # self.dataset_file = "imputedData.csv"

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

        self.selectors = [
            ('F-Regression', SelectKBest(score_func=f_regression)),
            ('Mutual Info Regression', SelectKBest(score_func=mutual_info_regression)),
            ('RFE', None)  # estimator is set later
        ]
        self.regressors = [
            ('Random Forest Regressor', RandomForestRegressor(n_estimators=100)),
            ('Ridge', Ridge()),
            ('Linear Regression', LinearRegression())
        ]

        self.X = None
        self.Y = None

    def main(self):
        self.load_dataset()
        self.scale()
        self.impute()

        print("Deleted features:")
        print(*DELETE_FEATURES, sep="\n")
        print("\n---\n")

        # fit the selectors and regressors (i.e. estimators or models)
        results = []
        for selector_name, selector in self.selectors:
            for regressor_name, regressor in self.regressors:
                if selector_name == "RFE":
                    selector = RFE(estimator=regressor, n_features_to_select=10)
                    # selector = RFECV(estimator=estimator, min_features_to_select=10, cv=5)
                    selected_features = self.select_rfe(selector)
                    regressor_score = self.train(selector, regressor)
                else:
                    selected_features = self.select(selector)
                    regressor_score = self.train(selector, regressor)
                # print the top selected features for this test
                print("{} selector with {}:".format(selector_name, regressor_name))
                print(*selected_features, sep="\n")
                print("{} 5 cross-fold average r2: {}".format(regressor_name, regressor_score))
                print("\n")
                results.append((selector_name, regressor_name, selected_features, regressor_score))
            print("---\n")

        # combine results from tests and print; first combine by selector
        selector_features = []
        for selector_name, selector in self.selectors:
            selector_feature_list = [r[2] for r in results if r[0] == selector_name]
            print("{} ranked results:".format(selector_name))
            print("\n")
            reg_results = compare_lists.get_common_items(*selector_feature_list)
            selector_features.append((selector_name, reg_results))

        # combine total results
        print("---COMBINED TOTAL---\n")
        compare_lists.get_total_items(*[r[1] for r in selector_features])

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
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

    def impute(self):
        if self.imputer:
            self.X = pd.DataFrame(self.imputer.fit_transform(self.X), columns=self.X.columns)

    def select(self, selector):
        selector.fit(self.X, self.Y)
        feature_dictionary = {key: value for (key, value) in zip(selector.scores_, self.X.columns)}
        return list(reversed(sorted(feature_dictionary.items())))[:10]

    def select_rfe(self, selector: RFE):
        selector = selector.fit(self.X, self.Y)
        features = []
        for bool, rank, feature in zip(selector.support_, selector.ranking_, self.X.columns):
            if bool:
                features.append((rank, feature))
        return list(reversed(sorted(features)))

    def train(self, selector, regressor):
        x_train = selector.transform(self.X) if selector else self.X
        return np.average(cross_val_score(regressor, x_train, self.Y, cv=5, scoring="r2"))


if __name__ == '__main__':
    FeatureSelector().main()
