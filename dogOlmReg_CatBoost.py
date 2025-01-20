# Kategorik değişkenler ile otomatik olarak mücadele edebilen, hızlı başarılı bir diğer GBM türevi

# kategorik değişken desteği
# hızlı ölçeklenebilir GPU desteği
# daha başarılı tahminler
# hızlı train ve hızlı tahmin
# rusyanın ilk açık kaynak kodlu, başarılı ML çalışması

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
from catboost import CatBoostRegressor

from CARTregresyon import y_train

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "NewLeague", "Division"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N","Division_W", "NewLeague_N"]]], axis=1)
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# model oluşturma
cat_model = CatBoostRegressor().fit(x_train, y_train)

# hata
y_pred = cat_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

print(cat_model.get_params())
print(cat_model.learning_rate_)


# model tuning
# iterations, ağaç sayısıdır ya da fit edilecek model sayısıdır
cat_params = {"iterations":[200,500],
              "learning_rate": [0.01,0.1],
              "depth": [3,6]}

cat_params
cat_model = CatBoostRegressor()

cat_cv_model = GridSearchCV(cat_model, cat_params, cv= 5, n_jobs=-1, verbose=2).fit(x_train, y_train)

# en iyi modeli çağırıp, tuned de bu en iyi modelin değerlerini kullanırız
cat_cv_model.best_params_
cat_tuned = CatBoostRegressor(depth=3, iterations=200, learning_rate=0.1)

y_pred= cat_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))