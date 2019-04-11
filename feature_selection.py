import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
import time

delete_features = ['Life Expectancy', 'Life expectancy at birth (years)',
                   'Use of basic sanitation services % (rural) 2015',
                   # 'urban use of basic sanitation services (%)',
                   'Use of basic sanitation services % (urban) 2015',
                   'Use of basic drinking water services % (rural) 2015.1',
                   'Use of basic drinking water services % (rural) 2015',
                   'Use of basic drinking water services % (urban) 2015',
                   'Reported Maternal mortality ratio',
                   # 'adolescent proportion of total population (%)',
                   # 'Total Fertility Rate',
                   ]

import warnings

warnings.filterwarnings("ignore")

country_to_region_map = {'Afghanistan': 'Least developed countries', ' Angola': 'Least developed countries',
                         ' Bangladesh': 'Least developed countries', ' Benin': 'Least developed countries',
                         ' Bhutan': 'Least developed countries', ' Burkina Faso': 'Least developed countries',
                         ' Burundi': 'Least developed countries', ' Cambodia': 'Least developed countries',
                         ' Central African Republic': 'Least developed countries', ' Chad': 'Least developed countries',
                         ' Comoros': 'Least developed countries',
                         ' Democratic Republic of the Congo': 'Least developed countries',
                         ' Djibouti': 'Least developed countries', ' Eritrea': 'Least developed countries',
                         ' Ethiopia': 'Least developed countries', ' Gambia': 'Least developed countries',
                         ' Guinea': 'Least developed countries', ' Guinea-Bissau': 'Least developed countries',
                         ' Haiti': 'Least developed countries', ' Kiribati': 'Least developed countries',
                         ' Lao People’s Democratic Republic': 'Least developed countries',
                         ' Lesotho': 'Least developed countries', ' Liberia': 'Least developed countries',
                         ' Madagascar': 'Least developed countries', ' Malawi': 'Least developed countries',
                         ' Mali': 'Least developed countries', ' Mauritania': 'Least developed countries',
                         ' Mozambique': 'Least developed countries', ' Myanmar': 'Least developed countries',
                         ' Nepal': 'Least developed countries', ' Niger': 'Least developed countries',
                         ' Rwanda': 'Least developed countries', ' Sao Tome and Principe': 'Least developed countries',
                         ' Senegal': 'Least developed countries', ' Sierra Leone': 'Least developed countries',
                         ' Solomon Islands': 'Least developed countries', ' Somalia': 'Least developed countries',
                         ' South Sudan': 'Least developed countries', ' Sudan': 'Least developed countries',
                         ' Timor-Leste': 'Least developed countries', ' Togo': 'Least developed countries',
                         ' Tuvalu': 'Least developed countries', ' Uganda': 'Least developed countries',
                         ' United Republic of Tanzania': 'Least developed countries',
                         ' Vanuatu': 'Least developed countries', ' Yemen': 'Least developed countries',
                         ' Zambia': 'Least developed countries'}

# data = pd.read_csv("Dataset in tabs - All Features.csv", index_col=0)
# data = data.fillna(value=-1)  # FIXME: specify missing value
# X = data[data.columns[1:]]
# Y = data[data.columns[0]]

data = pd.read_csv("imputedData.csv", index_col=0)
data.drop(columns=delete_features, axis=1, inplace=True)
data = data.fillna(value=-1)  # FIXME: specify missing value
X = data[data.columns[2:]]
Y = data[data.columns[1]]

# Update the models' hyperparameters here
models = {'DecisionTree': DecisionTreeRegressor(), 'KNN': KNeighborsRegressor(), 'NaiveBayes': GaussianNB(),
          'MLP': MLPRegressor(batch_size=1),
          #   'SVM': SVR(),
          'RandomForest': RandomForestRegressor(n_estimators=100),
          'Ridge': Ridge(),
          'LinearRegression': LinearRegression()}

scoring = ['neg_mean_squared_error', 'r2']

estimator = DecisionTreeRegressor()

full_results_r2 = []
full_results_RMSE = []
full_start = time.time()

scaler = MinMaxScaler()
scaler = scaler.fit(X, Y)
X_scaled = scaler.transform(X)
X_headers = list(X)

trial_data = {}

# TODO: change 50 to the max number of features to test with
for num_features in range(50):
    # for num_features in range(len(X.columns)):
    sub_results_r2 = []
    sub_results_RMSE = []

    rfecv = None
    X_transformed = []
    if num_features != 0:
        # TODO: change RFECV() to whatever feature selection method
        rfecv = RFECV(estimator, min_features_to_select=num_features)
        rfecv = rfecv.fit(X_scaled, Y)
        num_features_selected = rfecv.n_features_

        feature_rankings = rfecv.ranking_
        selected_features = np.argwhere(feature_rankings == 1).flatten()

        selected_feature_names = []
        for index in selected_features:
            selected_feature_names.append(X_headers[index])
        trial_data[num_features] = {"num_features_selected": num_features_selected,
                                    "feature_indices": selected_features, "feature_names": selected_feature_names}

        X_transformed = rfecv.transform(X_scaled)

    else:
        rfecv = None

    for name, model in models.items():
        # To add a preprocessor, just add it to the pipeline here. You can also pull them out and put them earlier if you
        # want to edit some of the hyperparams
        # ,
        if num_features == 0:
            sub_results_r2.append(name)
            sub_results_RMSE.append(name)
            continue

        pipe = Pipeline([
            # ('norm', MinMaxScaler()), ('RFE', RFECV(estimator, min_features_to_select=num_features)),
            ('model', model)])
        result = cross_validate(pipe, X_transformed, Y, cv=5, scoring=scoring)
        sub_results_r2.append(np.mean(result['test_r2']))
        sub_results_RMSE.append(np.mean(np.sqrt(-result['test_neg_mean_squared_error'])))

        print("finished %s with %s features" % (name, num_features))

    full_results_r2.append(sub_results_r2)
    full_results_RMSE.append(sub_results_RMSE)

full_end = time.time()
full_results_r2_np = np.array(full_results_r2)
full_results_RMSE_np = np.array(full_results_RMSE)
r2_np = np.array(full_results_r2[1:])
rmse_np = np.array(full_results_RMSE[1:])

best_num_features_r2 = np.argmax(r2_np, axis=0)
best_num_features_rmse = np.argmin(rmse_np, axis=0)

print("\n######## R2 ########")
for i in range(len(best_num_features_r2)):
    print("Model: %s" % full_results_r2[0][i])
    print("\tBest Accuracy: %s" % r2_np[best_num_features_r2[i]][i])
    print("\tNum Features Selected: %s" % trial_data[best_num_features_r2[i] + 1]['num_features_selected'])
    print("\tFeature Names")
    print(trial_data[best_num_features_r2[i] + 1]['feature_names'])
    print("\n")

print("\n######## RMSE ########")
for i in range(len(best_num_features_rmse)):
    # print("Model: %s - Best Num Features: %s - Accuracy: %s" % (full_results_RMSE[0][i], best_num_features_rmse[i] + 1, rmse_np[best_num_features_rmse[i]][i]))
    print("Model: %s" % full_results_RMSE[0][i])
    print("\tBest Accuracy: %s" % rmse_np[best_num_features_r2[i]][i])
    print("\tNum Features Selected: %s" % trial_data[best_num_features_rmse[i] + 1]['num_features_selected'])
    print("\tFeature Names")
    print(trial_data[best_num_features_rmse[i] + 1]['feature_names'])
    print("\n")

# print(trial_data)
print("Elapsed time: %s" % (full_end - full_start))
