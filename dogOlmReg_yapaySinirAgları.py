#insan beyninin bilgi işleme şeklini referans alan sınıflandırma
# ve regresyon problemleri için kullanılabilen kuvvetli makine öğrenmesi
# algoritmalarından birisidir.

# amaç en küçük hata ile tahmin yapabilecek katsayılara erişmektir

# sinir hücresi; dendrit, akson, soma ve snapsis den oluşur
#  dentrinin görevi, gelen sinyalleri somaya iletmektir
# somanın görevi, dentrilerden gelen bilgileri toplamaktır
# aksonun görevi, toplanan bilgileri hücrelere iletilmesi için aktarım yapar
# snapsis'in görevi bilgileri dönüşüme uğratmaktır ve daha sonra bilgileri hücrelere iletmektir


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

df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division","NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# model tahmin, standartlaştırma işlemi gerçekleştirecek
# yapay sinir ağları, homojen veri setlerinde daha iyi çalışır.
# standartlaştırma işlemi gerçekleştirdikten sonra kullanılmalı


# StandardScaler, veri setindeki her bir özelliğin (değişkenin) ortalamasını 0 ve standart sapmasını 1 yapmak
# için standartlaştırma (veya normalizasyon) işlemi uygular.
scaler = StandardScaler()

scaler.fit(x_train)
# Ölçeklendirilmiş veri, genellikle modelin daha hızlı öğrenmesini ve daha iyi performans göstermesini sağlar.

# transform işlemi, veriyi uygun bir ölçeğe dönüştürür. Hangi tür ölçekleme kullandığınıza bağlı olarak:
    # StandardScaler: Veriyi ortalaması 0 ve standart sapması 1 olacak şekilde ölçekler.
x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)

# çok katmalı model
mlp_model = MLPRegressor().fit(x_train_scaled, y_train)

mlp_model

mlp_model.predict(x_test_scaled)[0:5]
y_pred = mlp_model.predict(x_test_scaled)

np.sqrt(mean_squared_error(y_test, y_pred))

# model tuning
# hidden_layer_sizes: gizli katman sayısı
mlp_params = {"alpha": [0.1,0.01,0.02,0.001,0.0001],
              "hidden_layer_sizes":[(10,20), (5,5),(100,100)]}
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose=2, n_jobs=-1).fit(x_train_scaled, y_train)

mlp_cv_model.best_params_

# final
mlp_tuned = MLPRegressor(alpha= 0.02,hidden_layer_sizes = (100,100)).fit(x_train_scaled, y_train)

y_pred = mlp_tuned.predict(x_test_scaled)

np.sqrt(mean_squared_error(y_test, y_pred))

