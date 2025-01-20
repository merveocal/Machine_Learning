#topluluk öğrenme yöntemleri: birden fazla algoritmanın ya da birden fazla
# ağacın bir araya gelerek toplu bir şekilde öğrenmesi ve tahmin etmeye çalışmasıdır


#Bagging: temeli bootstrap yöntemi ile oluşturulan birden fazla karar ağacının ürettiği tahminlerin bir araya getirilerek değerlendirilmesine dayanır
#çalışma prensibinin kilit noktası, bootstrap rastegele örnekleme yöntemidir

# Bagging yöntemi
    # hata kareler ortalamasının karekökü değerini düşürür
    # doğru sınıflandırma oranını arttırır
    # varyansı düşürür ve ezberlemeye karşı dayanıklıdır


# Random Forests
    # temeli birden çok karar ağacın ürettiği tahminlerin bir araya getirilerek değerlendirilmesine dayanır
    # gözlem seçme işlemin de rassalık getirdi

# ağaçlar için gözlemler bootstrap rastgele örnek seçim yöntemi ile değişkenler random subspace yöntemi ile seçilir
# #karar ağacnın her bir düğümünde en iyi dallara ayırıcı (bilgi kazancı) değişken tüm değişkenler arasından rastegele seçilen daha az sayıdaki değişken arasından seçilir
# #ağaç oluşturmada veri setinin 2/3 'ü kullanılır. dışarıda kalan veri ağaçlarının prformans değerlendirmesi ve değişken öneminin belirlenmesi için kullanılır
## her düğüm noktasında rastegele değişken seçimi yapılır (regresyon p/3, sınıflama da karekök p)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
from cokluDogrusalRegresyon import y_train
# from sklearn.preprocessing import scale bunda hata veriyor
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


df = pd.read_csv("./Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis = 1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# model tahmin
rf_model = RandomForestRegressor(random_state=42).fit(x_train, y_train)
rf_model


y_pred = rf_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))



# Model Tuning işlemi
# max_features: bölünmelerde göz önünde bulundurulası gereken değişken sayısını ifade eder
rf_params = {"max_depth": [8,10],
             "max_features":[2,10],
             "n_estimators": [200,500],
             "min_samples_split": [10,80,10]}

rf_cv_model = GridSearchCV(rf_model,rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(x_train, y_train)
rf_cv_model

rf_cv_model.best_params_