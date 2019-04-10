import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

SELECTED_FEATURES = ['2015 Adjusted Maternal mortality ratio',
                     '2016 Crude Birth Rate',
                     'urban use of basic sanitation services (%)',
                     'Use of basic drinking water services % (total) 2015',
                     'Number per 200 population 2016 internet users',
                     'adolescent birth rate',
                     '2015 Maternal mortality ratio: Lifetime risk of maternal death (1 in:)',
                     'Ratio of Male:Female 2016 Under-5 Mortality Rates',
                     'Total Fertility Rate',
                     'Total adult literacy rate (%)',
                     'adolescent proportion of total population (%)',
                     'rural use of basic sanitation services (%)',
                     'Delivery care (%): Skilled birth attendant']

delete_features = ['Life Expectancy',
                   'Life expectancy at birth (years)',
                   'Use of basic sanitation services % (rural) 2015',
                   'Use of basic sanitation services % (urban) 2015',
                   'Use of basic drinking water services % (rural) 2015.1',
                   'Use of basic drinking water services % (rural) 2015',
                   'Use of basic drinking water services % (urban) 2015',
                   'Reported Maternal mortality ratio',
                   ]

models = {'DecisionTree': DecisionTreeRegressor(),
          'KNN': KNeighborsRegressor(),
          'MLP': MLPRegressor(batch_size=1),
          #   'SVM': SVR(),
          'RandomForest': RandomForestRegressor(n_estimators=100),
          'Ridge': Ridge(),
          'LinearRegression': LinearRegression()}


data = pd.read_csv("imputedData.csv", index_col=0)
data.drop(columns=delete_features, axis=1, inplace=True)

X = data[data.columns[2:]]
# X = data[SELECTED_FEATURES]
Y = data[data.columns[1]]

scaler = MinMaxScaler()
scaler = scaler.fit(X, Y)
X_scaled = pd.DataFrame(scaler.transform(X))

X_scaled = X_scaled.fillna(value=-1)

scoring = ['neg_mean_squared_error', 'r2']

for name, model in models.items():
    # To add a preprocessor, just add it to the pipeline here. You can also pull them out and put them earlier if you
    # want to edit some of the hyperparams
    result = cross_validate(model, X_scaled, Y, cv=5, scoring=scoring)
    print(f"{name}\tr2: {np.mean(result['test_r2']):.3f}\tRMSE: {np.mean(np.sqrt(-result['test_neg_mean_squared_error'])):.3f}")
