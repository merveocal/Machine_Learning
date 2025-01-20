import numpy as np
import pandas as pd
from mpl_toolkits.axisartist.grid_finder import GridFinder

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



#  Amaç. iki sınıf arasındaki ayrımın optimum olmasını sağlayacak hiper-düzlemi bulmaktır
#  sınıflandırma problemleri için ortaya çıkmıştır


df = pd.read_csv("./diabetes.csv")
df = df.dropna()
df.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# MODEL & TAHMİN
svm_model = SVC(kernel = "linear").fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
accuracy_score(y_test, y_pred)

print(svm_model.get_params())


# MODEL TUNING

# boş model nesnesi oluşturma
svm = SVC()

#  "c", ceza parametresi
svm_params = {"C": np.arange(1,10), "kernel":["linear", "rbf"]}

svm_cv_model = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, verbose =2).fit(x_train, y_train)

svm_cv_model.best_score_

svm_cv_model.best_params_


# final modeli oluşturma
svm_tuned = SVC(C = 2, kernel= "linear").fit(x_train, y_train)
y_pred = svm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
