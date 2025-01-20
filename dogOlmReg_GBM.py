# adaboost'un sınıflandırma ve regresyon problemlerine kolayca uyarlanabilen genelleştirilmiş versionudur
# artıklar üzerine tek bir tahminsel model formunda olan modeller serisi kurulur

# mantığı, zayıf öğrencileiri bir araya getirip güçlü bir öğrenci ortaya çıkarmak fikrine dayanır

# kötü tahmin, gerçek değerler ile tahmin edile değerlerin farkının karelerinin alınması sonucunda ortaya çıkan  büyük değerlerir. kötü tahminde bulunan ağaçlar da zayıf tahmincilerdir

# Adaptive Boosting(AdaBoost) : zayfı sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturma fikrini hayata geçiren algoritmadır

#Gradient boosting tek bir tahminselmodel formunda olan modeller serisi oluşturur
# Seri içerisindeki bir model serideki bir önceki modelin tahmin artıklarının/hatalarının üzerine kurularak oluşturulur
# GBM diferansiyellenebilen herhangi bir kayıp fonksiyonunu optimizde edebilen Gradient descent algoritmasını kullanmakta
# GB bir çok temel öğrenci tipi (base kearner type) kullanılabilir

# cost fonksyionları ve link fonksiyonlar modifiye edilebilirer

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
dms = pd.get_dummies(df[["Division", "League", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "NewLeague", "Division"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 42)


# model kurma işlemi
gbm_model = GradientBoostingRegressor().fit(x_train, y_train)
# böyle yazınca hiperparametrelerini görüntüleyebiliyoruz
print(gbm_model.get_params())

# ilkel test hatası hesaplama
y_pred = gbm_model.predict(x_test)  #model tahmini yapmak
np.sqrt(mean_squared_error(y_test, y_pred)) #model tahmin sonucu hesaplama



# model tuning işlemi
gbm_params = {"learning_rate": [0.001, 0.1, 0.01],
              "max_depth": [3,5],
              "n_estimatrs": [100,200],
              "subsample": [1,0.5],
              "loss": ["ls", "las", "qauntile"]}

gbm_model = GradientBoostingRegressor().fit(x_train, y_train)

gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv = 10,n_jobs=-1, verbose= 2).fit(x_train, y_train)
gbm_cv_model.best_params_


gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1, loss = "lad",
                                      max_depth = 3,
                                      n_estimators=200,
                                      subsample = 1).fit(x_train, y_train)


