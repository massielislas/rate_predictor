import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Dataset in tabs - All Features.csv", index_col=0)
data = data.fillna(value=-1)

X = data[data.columns[1:]]
Y = data[data.columns[0]]

# Update the models' hyperparameters here
models = {'DecisionTree': DecisionTreeRegressor(), 'KNN': KNeighborsRegressor(), 'NaiveBayes': GaussianNB(),
          'MLP': MLPRegressor(), 'SVM': SVR(), 'RandomForest': RandomForestRegressor(n_estimators=100)}

scoring = ['neg_mean_squared_error', 'r2']

for name, model in models.items():
    # To add a preprocessor, just add it to the pipeline here. You can also pull them out and put them earlier if you
    # want to edit some of the hyperparams
    pipe = Pipeline([('norm', Normalizer()), ('PCA', PCA(n_components=1)), ('model', model)])
    result = cross_validate(model, X, Y, cv=5, scoring=scoring)
    print(f"{name}\tr2: {np.mean(result['test_r2']):.3f}\tRMSE: {np.mean(np.sqrt(-result['test_neg_mean_squared_error'])):.3f}")
