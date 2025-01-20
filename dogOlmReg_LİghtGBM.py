# Light GBM, XGBoost'un eğitim süresi performansını arttırmaya yönelik geliştirilen bir diğer GBM türüdür

#karar ağaçlarına dayanıyor
# daha performanslı
#Level-wise büyüme stratejisi yerine Leaf-wise büyüme stratejisi
# breadht-first search(BFS) yerine depth-first search(DFS)

import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League","Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "NewLeague", "Division"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# model ve tahmin

lgb_model = LGBMRegressor()
lgb_model.fit(x_train, y_train)

y_pred = lgb_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))



# model tuning

lgbm_params = {"learning_rate":[0.01,0.1,0.5,1],
               "n_estimators": [20,40,100],
               "max_depth": [1,2,3,4,5,6]}

lgbm_cv_model = GridSearchCV(lgb_model, lgbm_params, cv = 10, n_jobs = -1, verbose=2).fit(x_train, y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(learning_rate=0.1, max_depth=6, n_estimators=20).fit(x_train, y_train)

y_pred = lgbm_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

