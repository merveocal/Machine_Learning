import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.svm import SVR


# SVR (Support Vector Regression) modelinde, kernel parametresi,
# regresyon modelinin doğrusal mı yoksa doğrusal olmayan bir şekilde mi
# çalışacağını belirler.

# (tek cümle)
# Destek vektör regresyonu,  modelleme yönteminin amacı, bir
# marjın aralığına maksimum noktayı en küçük hata ile alabilecek bir
# doğru ya da eğri bulmaktır.


# (tek cümle)
# Gradient Boosting Machines, AdaBoost’un sınıflandırma ve regresyon
# problemlerine kolayca uyarlanabilen genelleştirilmiş versiyonudur.


df = pd.read_csv("./Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train_, y_test = train_test_split(x,y, test_size=0.25, random_state=42)


# model & tahmin
# svr modelini başlatırken, "linear" gibi bir string parametresi değil, doğr anahtar kelime parametresi olan "kernel'i kullanmanz gerekiyor
svr_model = SVR(kernel="linear").fit(x_train, y_train_)


# tahmin
svr_model.predict(x_train)
svr_model.predict(x_test)[0:5]

# katsayıyı getir
svr_model.intercept_


svr_model.coef_


# test hatası hesaplama
y_pred = svr_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))





# ######################
# model tuning
#########
svr_model = SVR(kernel="linear")
svr_model


svr_params = {"C": [0.1,0.5,1,3]}

# bir modelin hiperparametrelerini optimize etmek için kullanılan bir model seçimi ve parametre optimizasyonu aracıdır.
# bu aracın amacı, belirli bir modelin hiperparametrelerini belirli bir aralıkta taramak ve en iyi performansı sağlayan parametre kombinasyonunu bulmaktır
svr_cv_model = GridSearchCV(svr_model, svr_params, cv= 10).fit(x_train, y_train_)

svr_cv_model.best_params_

svr_cv_model = GridSearchCV(svr_model, svr_params, cv= 5, verbose = 2, n_jobs=-1).fit(x_train, y_train_)

svr_cv_model.best_params_


svr_tuned = SVR(kernel="linear", C = 0.5).fit(x_train, y_train_)

y_pred = svr_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

