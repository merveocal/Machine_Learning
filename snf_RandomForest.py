# temeli birden çok karar ağacının ürettiği tahminlerin bir araya getirilerek değerlendirilmesine dayanır
# bagging yöntemi: temeli bootstrap yöntemi ile oluşturulan birden fazla karar ağacının ürettiği tahminlerin bir araya getirilerekk değerlendirilmesine dayanır

# ağaçlar için gözlemler bootstrap rastegele örnek seçim yöntemi ile değişkenler random sunspace yöntemi ile seçilir

# karar ağacının her bir düğümünde en iyi dallara ayırıcı(bilgi kazancı) değişken tüm değişkenler arasından rastegele seçilen daha az sayıdaki değişken arasından seçilir

#ağaç oluşturmada veri setinin 2/3'ü kullanılır. dışarıda kalan veri ağaçların performans değerlendirmesi ve değişken öneminin belirlenmesi için kullanılır

# her düğüm noktasında rastegel değişken seçimi yapılır


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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("./diabetes.csv")
df = df.dropna()
y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)
df.head()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)


#  MODEL VE TAHMİN
rf_model = RandomForestClassifier().fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_test, y_pred)

print(rf_model.get_params())

# MODEL TUNING
rf = RandomForestClassifier()
rf_params = {"n_estimators":[10,20,50],
            "min_samples_split":[2,5,10],
             "max_features":[3,5,7]}

rf_cv_model = GridSearchCV(rf, rf_params, cv = 10, n_jobs=-1, verbose=2).fit(x_train, y_train)

rf_cv_model.best_params_


# final model
rf_tuned = RandomForestClassifier(min_samples_split=10, n_estimators=50, max_features=7).fit(x_train, y_train)

# test hatası hesaplama
y_pred = rf_tuned.predict(x_test)
accuracy_score(y_test, y_pred)



# değişken önem düzeyleri
print(rf_tuned.get_params())
rf_tuned.feature_importances_

feature_imp = pd.Series(rf_tuned.feature_importances_,
                        index= x_train.columns).sort_values(ascending=False)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel("değişken önem skorları")
plt.ylabel("değişkenler")
plt.title("değişken önem düzeyleri")
plt.show()

