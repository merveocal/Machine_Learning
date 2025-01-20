# amaç, hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır
# elasticNet L1 ve L2 yaklaşımları birleştirir

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from sklearn.linear_model import RidgeCV, LassoCV,  ElasticNet
from sklearn.linear_model import ElasticNetCV

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League","Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)


enet_model = ElasticNet().fit(x_train, y_train)
enet_model.coef_
enet_model.intercept_

# tahmin
enet_model.predict(x_train)

enet_model.predict(x_test)


# hata hesaplama
y_pred = enet_model.predict(x_test)

# hata kareler ortalaması karekökü
np.sqrt(mean_squared_error(y_test, y_pred))




#######################
# ElasticNet model tuning (model doğrulama)
########################

enet_cv_model = ElasticNetCV(cv = 10).fit(x_train, y_train)


# Elastik net modelinin, hiperpametrik alfa değeridir. Bu değer,
# modelin L1 ve L2 normlarının toplamını nasıl dengeleyeceğini belirler.
# Alfa değeri küçükse,L1 normunun etkisi artarken, büyükse L2 normunun etkisi artar.
enet_cv_model.alpha_

enet_cv_model.intercept_

enet_cv_model.coef_

enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(x_train, y_train)

enet_tuned

# hata hesaplama
y_pred = enet_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

