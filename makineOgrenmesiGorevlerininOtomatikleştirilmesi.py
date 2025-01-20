# import numpy as np
# # fonksiyon yazarak tekrarı azaltma
#
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])


# fonksiyon tanımlama
def compML(df, y, alg):
    # train-test ayrimi
    y = df[y]
    x_ = df.drop(["Salary", "League", "NewLeague", "Division"], axis=1).astype("float64")
    x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # modelleme
    model = alg().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    # Burada neden hata verdiğini çözemedim.aşağıda fonksiyonu çağırdığımda hata veriyor
    model_ismi = alg.__name__
    print(model_ismi, "modeli test hatası:" , RMSE)



# Yukarı da uzun uzun yazmak yerine tek bir sefer de hesaplayabiliyoruz
compML(df, "Salary", SVR)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

models = [LGBMRegressor, XGBRegressor, GradientBoostingRegressor,
          RandomForestRegressor, DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor,
          SVR]


for i in models:
    print(i, "Algoritmasının Test Hatası", compML(df,"Salary", i))


y = df["Salary"]
x_ = df.drop(["Salary","League", "NewLeague", "Division"], axis=1).astype("float64")
x = pd.concat([x_,dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)




































































