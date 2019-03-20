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

country_to_region_map = {'Afghanistan': 'Least developed countries', ' Angola': 'Least developed countries', ' Bangladesh': 'Least developed countries', ' Benin': 'Least developed countries', ' Bhutan': 'Least developed countries', ' Burkina Faso': 'Least developed countries', ' Burundi': 'Least developed countries', ' Cambodia': 'Least developed countries', ' Central African Republic': 'Least developed countries', ' Chad': 'Least developed countries', ' Comoros': 'Least developed countries', ' Democratic Republic of the Congo': 'Least developed countries', ' Djibouti': 'Least developed countries', ' Eritrea': 'Least developed countries', ' Ethiopia': 'Least developed countries', ' Gambia': 'Least developed countries', ' Guinea': 'Least developed countries', ' Guinea-Bissau': 'Least developed countries', ' Haiti': 'Least developed countries', ' Kiribati': 'Least developed countries', ' Lao People’s Democratic Republic': 'Least developed countries', ' Lesotho': 'Least developed countries', ' Liberia': 'Least developed countries', ' Madagascar': 'Least developed countries', ' Malawi': 'Least developed countries', ' Mali': 'Least developed countries', ' Mauritania': 'Least developed countries', ' Mozambique': 'Least developed countries', ' Myanmar': 'Least developed countries', ' Nepal': 'Least developed countries', ' Niger': 'Least developed countries', ' Rwanda': 'Least developed countries', ' Sao Tome and Principe': 'Least developed countries', ' Senegal': 'Least developed countries', ' Sierra Leone': 'Least developed countries', ' Solomon Islands': 'Least developed countries', ' Somalia': 'Least developed countries', ' South Sudan': 'Least developed countries', ' Sudan': 'Least developed countries', ' Timor-Leste': 'Least developed countries', ' Togo': 'Least developed countries', ' Tuvalu': 'Least developed countries', ' Uganda': 'Least developed countries', ' United Republic of Tanzania': 'Least developed countries', ' Vanuatu': 'Least developed countries', ' Yemen': 'Least developed countries', ' Zambia': 'Least developed countries'}

data = pd.read_csv("Dataset in tabs - All Features.csv", index_col=0)
data = data.fillna(value=-1)  # FIXME: specify missing value
X = data[data.columns[1:]]
Y = data[data.columns[0]]

# data = pd.read_csv("imputedData.csv", index_col=0)
# data = data.fillna(value=-1)  # FIXME: specify missing value
# X = data[data.columns[2:]]
# Y = data[data.columns[1]]

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
