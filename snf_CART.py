# Amaç, verş seti içerisindeki karmaşık yapıları  basit karar yapılarına dönüştürmektir

# heterojen veri setleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır

import numpy as np
import pandas as pd
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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./diabetes.csv")
df = df.dropna()
y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

df.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)



# MODEL VE TAHMİN AŞAMASI
cart_model = DecisionTreeClassifier().fit(x_train, y_train)
print(cart_model.get_params())

# test hatası hesaplama
y_pred = cart_model.predict(x_test)
accuracy_score(y_test, y_pred)



# MODEL TUNING AŞAMASI
cart = DecisionTreeClassifier()
cart_params = {"max_depth": [1,3,5],
               "min_samples_split": [2,3,5]}

cart_cv_model = GridSearchCV(cart, cart_params, cv=10, n_jobs=-1, verbose=2).fit(x_train, y_train)

cart_cv_model.best_params_

# final model
cart_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=3).fit(x_train, y_train)

y_pred = cart_tuned.predict(x_test)
accuracy_score(y_test, y_pred)



