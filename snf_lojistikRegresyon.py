#ön giriş: veri setisinde bağımlı değişkenin sınıflardan oluştuğu durumlarda kullanılan modelleme türüdür

#lojistik regresyon
# k-en yakın komuşu
# destek vektör makineleri
# yapay sinir ağlar
# cart


# LOJİSTİK REGRESON
# AMAÇ: sınıflandırma problemi için bağımlı ve bağımsız değişkenler arasındaki ilişkiyi tanımlayan doğrusal bir model kurmaktır
#çoklu doğrusal regresyonun sınıflandırma problemlerine uyarlanmış fakat ufak farklılıklara tabi tutulmuş bir versiyon olarak düşünebiliriz


# bağımlı değişken kategoriktir
# adını bağımlı değişkene uygulanan logit dönüşümden alır
# doğrusal regresyonda aranan varsyımlar burada aranmadığı için daha esnek kullanılabilirliği vardır

#bağımlı değişkenin 1 olarak tanımlanan değerinin gerçekleşme olasılığı hesaplanır. dolayısıyla bağımlı değişkenin alacağı değer ile ilgilenmez
# lojistik fonksiyonu sayesinde üretilen değerler 0-1 arasında olur


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
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



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv("./diabetes.csv")
df = df.dropna()
df.head()
# boş gözlemleri kaldırma



# model & tahmin
# outcome, bağımlı değişken
df["Outcome"].value_counts()
# describe, sayısal sütunlar için temel istatistik özetleri sağlar(min, count, std) gibi
#  T. ise dataframe'nin satır ve sütunlarını yer değiştiri. yani sütunlar satır olur, satırlarda sütun olur. bu describe çıktısını daha okunabilir hale getirmek için kullanılabilir

df.describe().T


y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)
x.head()



loj_model = LogisticRegression(solver = "liblinear").fit(x,y)

# katsayıyı verir
loj_model.intercept_

# bağımısz değişkenlere ilişkin katsayılara erişmiş oluruz
loj_model.coef_

# tahmin edilen değerler
loj_model.predict(x)[0:10]

y[0:10]

y_pred = loj_model.predict(x)
confusion_matrix(y, y_pred)

# doğruluk oranını verir
accuracy_score(y, y_pred)


print(classification_report(y, y_pred))

loj_model.predict_proba(x)[0:10]




# burda hata verdi. y'yi tek boyutlu hale getireceğim
y = y.ravel()
print(y.shape)

# roc eğrisine ilişkin kodlar
#  y, gerçek değerler. loj_model.predict(x), tahmin edilen değerler
logit_roc_auc = roc_auc_score(y, loj_model.predict(x))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label="AUC (area = %0.2f)" % logit_roc_auc)
plt.plot([0,1], [0,1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.1,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig("Log_ROC")
plt.show()



# LOJİSTİK REGRESYON MODEL TUNING
# model validationa işlemi yapılacaktır lojistik regresyon da
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# cross validation işlemini test hatasını hesaplamak içi yapacağım
print(x_train.shape)
print(y_train.shape)

loj_model = LogisticRegression(solver = "liblinear").fit(x_train, y_train)
y_pred = loj_model.predict(x_test)
print(accuracy_score(y_test, y_pred))

cross_val_score(loj_model, x_test, y_test, cv = 10).mean()



