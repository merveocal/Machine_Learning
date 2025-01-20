import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from basitDogrusalRegresyon import yeni_veri

matplotlib.use("TkAgg")
df = pd.read_csv("./Advertising.csv")
df = df.iloc[:, 1:len(df)]
df.head()

x = df.drop("sales", axis=1)
y = df["sales"]


x.head()
y.head()


# statsmodels ile model kurmak
import statsmodels.api as sm

lm = sm.OLS(y, x)
model = lm.fit()
model.summary()

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(x,y)

# sabit sayıya ulaşmak
model.intercept_

# bağımısz değişkenlerimize ilişkin katsayıya ulaşmak için (x.head(), kaç tane bağımısz değişken olduğunu kontrol etme)
model.coef_


sns.regplot(x=df[["TV"], df["radio"], df["newspaper"]], y =df["sales"], ci=None, scatter_kws={"color":"g", "s":9})

yeni_veri = [[30],[10],[40]]

yeni_veri = pd.DataFrame(yeni_veri).T
yeni_veri

model.predict(yeni_veri)

# hata kareler ortalaması
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, model.predict(x))
mse

# hata kareler ortalama karekök
import numpy as np
rmse = np.sqrt(mse)
rmse



# model tuning (model doğrulama)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
df = pd.read_csv("./Advertising.csv")
df.iloc[:, 1:len(df)]

x = df.drop("sales", axis=1)
y = df["sales"]

x.head()
y.head()


# sınama seti
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=99)

x_train.head()
x_test.head()
y_train.head()
y_test.head()


from sklearn.linear_model import LinearRegression
# linearRegression kullanırken hata vermemesini istiyorsak, "slearn.linear_model" 'i import etmemiz gerekiyor. üstteki komutu yaz
lm = LinearRegression()
model = lm.fit(x_train,y_train)


# hata kareler ortalamasını kullanmak için "numpy" kütüphanesini import et
import numpy as np
# x içindeki bağımsız değişkenleri kullanarak bir tahminde buluncak ve sonra da y de bu değerleri karşılaştıracak
from sklearn.metrics import mean_squared_error
# eğitim hatası
np.sqrt(mean_squared_error(y_train, model.predict(x_train)))


# test hatası
np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

# k katlı crossvalidation
from sklearn.model_selection import cross_val_score
cross_val_score(model, x_train, y_train, cv=10,scoring="neg_mean_squared_error")

#  k katlı cv ortalaması
np.mean(-cross_val_score(model, x_train, y_train, cv=10, scoring = "neg_mean_squared_error"))


# cv rmse
np.sqrt(np.mean(-cross_val_score(model, x_train, y_train, cv=10, scoring = "neg_mean_squared_error")))

# cross validation bize doğrulanmış bir hata verir
# cv rmse
np.sqrt(np.mean(-cross_val_score(model,x, y, cv= 10, scoring="neg_mean_squared_error")))


