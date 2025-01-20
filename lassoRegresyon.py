# düzenleleştirme yöntemi de denir
# Amaç, hata kareler toplamını minimize eden katsayıları, bu katsayılara bir ceza uygulayarak bulmaktr

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
# sonradan ekledim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from statsmodels.sandbox.regression.penalized import coef_restriction_diffbase

from cokluDogrusalRegresyon import x_train, y_train
from ridgeRegresyon import lamdalar1

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# y_train_scaled = scaler.fit_transform(y_train)  : bunu iptal ettim, yanlış

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


df = pd.read_csv("./Hitters.csv")
df = df.dropna()
# eksik verileri kaldırdı
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")

x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

df.head()

# 20 değişken, 263 gözlem birimi
df.shape


lasso_model = Lasso().fit(x_train_scaled, y_train)
lasso_model

# sabit sayı/kesişim noktası
lasso_model.intercept_

# katsayıyı aldı
lasso_model.coef_


# farklı lamda değerlerine karşılık katsayılar
lasso = Lasso()
coefs = []
# alphas = np.random.randint(0,1000,10)
alphas = 10**np.linspace(10,-2,100)*0.05
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(x_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")














#####################
# TAHMİN
#####################

lasso_model

lasso_model.predict(x_train)[0:5]
lasso_model.predict(x_test)[0:5]


# hata tespit
y_pred = lasso_model.predict(x_test)
# benimki daha fazla çıktı, yukarıda bir yerde hata yapmış olabilirim
np.sqrt(mean_squared_error(y_test, y_pred))

# Modelin açıklanabilirliğini ifade etmektedir. bağımısız değişkenlerin, bağımlı değişkende değişikliğin yüzde kaçını açıkladığı ifade etmektedir
r2_score(y_test, y_pred)




############################
# model tuning
##########################
lasso_cv_model = LassoCV(cv = 10, max_iter = 100000).fit(x_train, y_train)

lasso_cv_model = LassoCV(alphas = lamdalar1, cv = 10, max_iter= 100000).fit(x_train, y_train)
lasso_cv_model.alpha_

# 1.yöntem, böyle de yapılabilir
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(x_train, y_train)

# 2.yöntem
lasso_tuned = Lasso(alpha= lasso_cv_model.alpha_).fit(x_train, y_train)

y_pred = lasso_tuned.predict(x_test)
# hata kareler ortalaması karekökünü buluyorsun burada
np.sqrt(mean_squared_error(y_test, y_pred))


pd.Series(lasso_tuned.coef_, index= x_train.columns)
