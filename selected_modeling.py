import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

SELECTED_FEATURES = ['2015 Adjusted Maternal mortality ratio',
                     '2016 Crude Birth Rate',
                     'Number per 200 population 2016 internet users',
                     'Use of basic drinking water services % (total) 2015',
                     'Total Fertility Rate',
                     '2015 Maternal mortality ratio: Lifetime risk of maternal death (1 in:)',
                     'adolescent birth rate',
                     'urban use of basic sanitation services (%)',
                     'lower secondary school gross enrollment ratio',
                     'rural use of basic sanitation services (%)',
                     'Projected (2016-2030) Average annual growth rate of urban population (%)',
                     'rural stunting prevalence in children under 5 (moderate & severe) (%)',
                     'adolescent proportion of total population (%)',
                     'Lower secondary school participation Net attendance ratio 2011-2016 (female)',
                     'urban stunting prevalence in children under 5 (moderate & severe) (%)',
                     '2016 Crude Death Rate']


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

# baseline:
# data.drop(columns=delete_features, axis=1, inplace=True)
# X = data[data.columns[2:]]

# selected features:
X = data[SELECTED_FEATURES]

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
