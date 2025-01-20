# XGBoost, GBM'in hız ve tahmin performansını arttırman üzere optimize
# edilmiş; ölçeklenebilir ve farklı platformlara entegre edilebilir halidir
# I.   R, Python, Hadoop ve Scala ile kullanılabilir
# II.  Hızlıdır, ağaca dayalı bir modeldir
# III. Tahmin başarısı yüksektir
# IV. Bir çok uluslararası yarışmada kendini kanıtlamıştır
# V.  Ölçeklenebilirdir

import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "NewLeague", "Division"], axis= 1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)



# model tahmin
import xgboost
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

xgb = XGBRegressor().fit(x_train, y_train)
print(xgb.get_params())



y_pred = xgb.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))



# Model Tuning
xgb = XGBRegressor()

# XGBRegressor modelinin eğitiminde kullanılabilecel hiperparametrelerin değerlerini tanımlar.
xgb_params = {"learning_rate": [0.1,0.01,0.5],
              "max_depth": [2,3,4,5],
              "n_estimators": [100,200,300],
              "colsample_bytree": [0.4,0.7,1]}
# cv, kaç katlı çapraz doğrulama yapmak istediğini ifade eder
xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(x_train, y_train)

xgb_cv_model.best_params_

xgb_tuned = XGBRegressor(colsample_bytree = 0.4, learning_rate= 0.1, max_depth = 5, n_estimators=100).fit(x_train, y_train)

y_pred = xgb_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))