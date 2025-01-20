import numpy as np
import pandas as pd
import pandas as pnd

df = pnd.read_csv("./Advertising.csv")
df = df.iloc[:, 1:len(df)]
df.head()

df[:]

df.info()
import seaborn as sns
sns.jointplot(x = "TV",  y ="sales", data=df, kind = "reg")



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# CSV dosyasını yükleme

df = pd.read_csv("./Advertising.csv")
df = df.iloc[:,1:len(df)]

# Veriyi kontrol etme
print(df.head())

# jointplot ile veri görselleştirme
sns.jointplot(x="TV", y="sales", data=df, kind="reg")
plt.show()  # Grafik gösterme

df.head()


import seaborn as sns

sns.jointplot(x="TV", y = "sales", data = df, kind="reg")

# anaconda kullandığım için sckitlearn kütüphanesini ekleyeceğim

from sklearn.linear_model import LinearRegression
x = df[["TV"]]
type(x)

y = df[["sales"]]

# model nesnesi oluşturma işlemi
reg = LinearRegression()

#fit, modeli kur
model = reg.fit(x,y)
str(model)
dir(model)


model.intercept_
model.coef_

# modelin skorunu ifade ediyor,"rkare"
model.score(x,y)


# scikit-learn kütüphanesi, makine öğrenmesinde kullanılan bir kütüphanedir


# bu model ile tahmin etme işlemi
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
df = pd.read_csv("./Advertising.csv")
df.iloc[:,1:len(df)]
g = sns.regplot(x=df["TV"], y=df["sales"], ci= None, scatter_kws ={'color': 'r', 's':9})
g.set_title("model denklem: sales = 7.03 + TV*0.05")
g.set_ylabel("satış sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-50,310)
plt.ylim(bottom=-5)



Sales = 7.03 + 0.04*df["TV"]
df

reg = LinearRegression()
# tahmin etme işlemidir aşağıdaki
model.predict([[165]])

# tahmin etme işlemidir
model.intercept_ + model.coef_ *165

yeni_veri = [[5],[15],[30]]
model.predict(yeni_veri)

model.predict([[450]])








# ARTIKLAR
# MSE: hata kareler ortalaması
# RMSE: Hata Kareler Ortalamasının Karekökü

y.head()
model.predict(x)[0:6]

gercek_y = y[0:10]
tahmin_edilen_y = pd.DataFrame(model.predict(x)[0:10])
hatalar = pd.concat([gercek_y, tahmin_edilen_y], axis=1)
hatalar.columns=["gercek_y", "tahmin_edilen_y"]
hatalar["hata"] = hatalar["gercek_y"]- hatalar["tahmin_edilen_y"]
# hata kareler ortalamasının karekökü, aşağıdadır
hatalar["hata_kareler"] = hatalar["hata"] **2
import numpy as np
# hata kareler ortalaması
np.mean(hatalar["hata_kareler"])