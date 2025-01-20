
# ADABOOSTun sınıflandırma ve regresyon parametrelerine kolayca uyarlanabilen genelleştirilmiş versiyonudur
# artıklar üzerine tek bir tahminsel modell formunda olan modeller serisi kurulu
# gradient boosting tek bir tahminsel model formunda olan modeller serisi oluşturur
#seri içerisindeki bir model serideki bri önceki modelin tahmin artıklarının üzerinde kurularak(fit) oluşturulur
#GBM diferansiyellenebilen herhangi bir kayıp fonksiyonunu optimize edebilen Gradient descenr algoritmasını kullanmakta
#gbm bir çok temel öğrenci tipi kullanılabilir
# cast fonksiyonları ve link fonksiyonları modeifiye edilebilirler


import numpy as np
import pandas as pd
from adodbapi.ado_consts import adCurrency
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sınıflandırmaModelleri.snf_yapaySinirAgları import y_train

df = pd.read_csv("./diabetes.csv")
df = df.dropna()
y = df["Outcome"]
x = df.drop(["Outcome"], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)
df.head()

# MODEL & TAHMİN
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier().fit(x_train, y_train)

#test hatası
y_pred = gbm_model.predict(x_test)
accuracy_score(y_test, y_pred)


# MODEL TUNING
gbm = GradientBoostingClassifier()
print(gbm.get_params())

gbm_params = {"learning_rate": [0.1,0.01,0.05],
              "n_estimators":[100,300,500],
              "max_depth":[2,3,5,8]}

gbm_cv_model = GridSearchCV(gbm, gbm_params, cv= 10, n_jobs=-1, verbose=2)

gbm_cv_model.best_params_

# final model
gbm_tuned = GradientBoostingClassifier(learning_rate=0.01,
                                       max_depth=5,
                                       n_estimators=500).fit(x_train, y_train)


gbm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)