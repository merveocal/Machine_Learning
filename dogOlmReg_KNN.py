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

# uyarı mesajlarıyla karşılaşmak istemiyorsan aşağıdakileri kullanabilirsin
from warnings import filterwarnings
filterwarnings("ignore")


# KNN
df = pd.read_csv("./Hitters.csv")
# satır ve sütunlardaki eksik gözlemleri siliyor
df = df.dropna()

# one hot encoding işlemi yapıyor aşağıdaki kodda
dms = pd.get_dummies(df[["Salary", "League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", 'NewLeague_N']]], axis=1)
# aynı bölmenin tekrarlanabilir olmasını sağlamak için "random_state" parametresi kullanılır.
x_train, x_test, y_train_, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

# model
# fit() fonksiyonu, modelin eğitilmesi için kullanılır. Yani, model, verilen eğitim verilerini
# (x_train) ve hedef etiketleri  (y_train_) kullanarak öğrenme işlemi yapar.
knn_model = KNeighborsRegressor().fit(x_train, y_train_)

# yukarıdaki işlemi şöyle de yapabilirdim
# knn_model = KNeighborsRegressor()
# knn_model.fit(x_train, y_train_)

knn_model



# komşu sayısını verir
knn_model.n_neighbors

#  komşular arasındaki mesafeleri hesaplayarak tahminlerde bulunur,
#  bu yüzden mesafe ölçütü önemli bir rol oynar.
knn_model.metric

# model nesnesi içerisinden alınabilecek değerleri vermiştir.
dir(knn_model)


# tahmin etme işlemini gerçekleştirir
knn_model.predict(x_test)[0:5]

y_pred = knn_model.predict(x_test)

# ilkel test hatası
np.sqrt(mean_squared_error(y_test, y_pred))





# k en yakın komşu model tuning
knn_model

RMSE = []

for k in range(10):
    k = k+ 1
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train_)
    y_pred = knn_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k=", k, " için rmse değeri", rmse)



# GridSearchCV: kullanacağımız makine algoritmalarında belirlemeye çalıştığımız hiperparametrelerin
# değerlerini belirlemek için bir fonksiyondur


knn_params = {"n_neighbors": np.arange(1,30,1)}

knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv= 10).fit(x_train, y_train_)

knn_cv_model.best_params_


# final modeli oluşturma
knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(x_train, y_train_)
y_pred = knn_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,  y_pred))

