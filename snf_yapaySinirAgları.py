# insan beyninin bilgi işleme şeklini referans alarak sınıflandırma ve regresyo problemleri için
#kullanılabilen kuvvetli makine öğrenmesi algoritmalarından birisidir

# amaç, en küçük hata ile tahmin yapabilecek katsayılara erişmektir

# Amaç, gerçek değerler ile algoritma ile tahmin edilmiş değerler arasındaki farkları minimuma indirmeye çalışmaktır

import numpy as np
import pandas as pd

from DograusalRegresyonModelleri.elasticNetRegresyon import x_train
from DogrusalOlmayanRegresyonModelleri.yapaySinirAgları import mlp_cv_model
from sınıflandırmaModelleri.lojistikRegresyon import y_train

pd.set_option("display.max_columns", None)
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


df = pd.read_csv("./diabetes.csv")
df = df.dropna()
df.head()

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.30, random_state=42)

df.head()

#  MODEL & TAHMİN

# model
mlp_model = MLPClassifier().fit(x_train, y_train)
mlp_model.coefs_

# ?mlp_model

# log-loss : gerçek değerler ile tahmine edilen değerler arasındaki farklara ilişkin optimizayon işlemi yapar
# hidden_layer_sizes
#  activation, relu--> doğrusal problemler için kullanılmaktedır. logistic yani sigmoid fonksiyonu kullanılır sınıflandırma da
# solver : modelimizdeki kullandığımızdaki aırlıkları optimize etmek için kullanılır. ön tanımlı değer "adan", büyük veris setlerinde dah aiyi çalışır. küçük veri setlerinde "lbfgs" kullanılır daha çok
#  alpha: ridge ve lasso da kullandığımız ceza terimiydi

#  test setimize ilişkin hata
y_pred = mlp_model.predict(x_test)
accuracy_score(y_test, y_pred)


# MODEL TUNING
# hidden_layer_sizes:  gizli katman sayısı
#activation: ön tanımlı olan "relu(regresyona ait olduğu için) kullanılmayacak onun yerine logistic(sınflandırmaya ait) olan kullanılacak
#  solver: ağırlık optimizasyonu için kullanılacak. ön tanımlı değeri "adan" ama "lgbfs" kullanacağız
#  alpha: ceza terimi düzenleştirme terimi

mlpc_params = {"alpha": [0.1,0.01,0.03],
               "hidden_layer_sizes": [(10,10),(100,100,100),(3,5)]}

mlpc = MLPClassifier(solver = "lbfgs")
mlp_cv_model = GridSearchCV(mlpc, mlpc_params, cv = 10, n_jobs=-1, verbose =2).fit(x_train, y_train)

mlp_cv_model

mlp_cv_model.best_params_


# final modeli
# mlpc_tuned aşamasında hata aldım bu hatayı "max_iter" değerini eklediğimde çözdüm.
# bu hatayı vermesinin sebebi: konverjans hatası aldığını belirtiyor bu da modelin verile iterasyon sayısında optimal çözümü bulamadığını gösterir.
# bu tür bir sorun, modelin yeterli sayıda iterasyonla eğitim almadığı veya verilerin uygın şekilde ölçeklenmediği durumlarda ortaya çıkar
mlpc_tuned = MLPClassifier(solver = "lbfgs", alpha = 0.03, hidden_layer_sizes=(10,10), max_iter=2000).fit(x_train, y_train)
y_pred = mlpc_tuned.predict(x_test)
accuracy_score(y_test, y_pred)

