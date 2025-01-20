# Amaç, hata kareler toplamını minimize eden katsayıları, bu katsayılara bir ceza uygulayarak bulmaktr


import numpy as np
import pandas as pd
# tüm sütunları getir dedim burada
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
df.head(3)
# pandas dataframe deki kategorik sütunları "one-hot encoding" yöntemiyle dönüştürmek için kullanılır
# one-hot encoding: her bir kategori için ayrı bir sütun oluşturur ve o kategoriye ait satırda 1, diğerlerinde 0 olacak şekilde veriyi dönüştürür
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])


y = df["Salary"]

# astype metot, bir pandas dataframe veya series veri tiplerini dönüştürmek için kullanılır
x_ = df.drop(["Salary", "League","Division", "NewLeague"], axis=1).astype("float64")

x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)

# train_test_split fonksiyonunu kullanarak veri setini eğitim ve test olmak üzerie ikiye ayırır
# aşağıdak test_size: 0.25, test setinin boyutunu belirtir. veri setini %25'i test setine %75'i eğitim setine ayrılır
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

df.head(3)

# toplam satır ve sütun sayısını döner
df.shape
# scikit-learn kütüphanesinde , ridge bir sınıftır
# alpha: ridge regresyonundaki "ceza teriminin" büyüklüğünü kontrol eder
    # daha büyük bir alpha deeri, daha güçlü bir regülarizasyon(katsayıların küçülmesi) uygular
    # daha küçük bir alpha değeri, regülarizasyon azaltır ve model standart doğrusal regresyona yaklaşır
ridge_model = Ridge(alpha=1).fit(x_train,  y_train)
ridge_model

# katsayılar
ridge_model.coef_

# sabit sayılar
ridge_model.intercept_


np.linspace(10,-2,100)
lambdalar = 10** np.linspace(10,-2,100)*0.5

lambdalar

# model nesnesi oluşturdu
ridge_model = Ridge()

katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(x_train, y_train)
    #coef_ :katsayı , intercept_: kesişim noktası
    katsayilar.append(ridge_model.coef_)

katsayilar

ax = plt.gca()
ax.plot(lambdalar, katsayilar)
ax.set_xscale("log")




####################
# Ridge regresyon modeli ile tahmin işlemi
######################

# predict:  tahmin yapmada kullanılır
# eğitim setinin bağımsız değişkenlerini değerlerini girerek bağımlı değişkenleri tahmin etmeye çalıştık
ridge_model = Ridge().fit(x_train,y_train)

# aşağıda x_eğitim verilerini kullanarak bir tahmin yaparak y_pred 'e atadık
y_pred = ridge_model.predict(x_train)

# train hatası
rmse = np.sqrt(mean_squared_error(y_train, y_pred ))
rmse


np.sqrt(np.mean(-cross_val_score(ridge_model,x_train, y_train, cv = 10, scoring="neg_mean_squared_error")))


# test hatası
y_pred = ridge_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse





###################################
# model tuning (model doğrulama)
###################################



# aşağıdaki kod, fit(x_train, y_train) bu adım da, model katsayıları ve kesişim değeri öğrenir
ridge_model = Ridge(1).fit(x_train, y_train)

# test verisi üzerindeki tahminler aşağıdaki kod ile görüntülenir
y_pred = ridge_model.predict(x_test)

# y_test, bağmlı değişken, y_pred: bağımsız değişkenin bağımlı değişkeni tahmin etmesi
np.sqrt(mean_squared_error(y_test, y_pred))
np.random.randint(0,100,10)
lamdalar1 = np.random.randint(0,1000,100)
lamdalar1

lamdalar2 = 10**np.linspace(10,-2,100)*0.5
lamdalar2

ridgecv = RidgeCV(alphas=lamdalar2, scoring="neg_mean_squared_error", cv=10 )
ridgecv.fit(x_train, y_train)

# içindeki optimum parametreyi alabilmek için
ridgecv.alpha_


# final modeli
ridge_tuned = Ridge(alpha=ridgecv.alpha_).fit(x_train, y_train)

y_pred = ridge_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))




# lamdalar1 için

ridgecv = RidgeCV(alphas=lamdalar1, scoring="neg_mean_squared_error", cv=10 )
ridgecv.fit(x_train, y_train)
ridgecv.alpha_

# final modeli
ridge_tuned = Ridge(alpha=ridgecv.alpha_).fit(x_train, y_train)

y_pred = ridge_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

