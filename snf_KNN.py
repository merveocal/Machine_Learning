# gözlemlerin birbirine olan benzerlikleri üzerinden tahmin yapılır

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


df = pd.read_csv("./diabetes.csv")
df = df.dropna()

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)
# random_state, her zaman 42 ye böler
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)



# MODEL & TAHMİN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_model = KNeighborsClassifier().fit(x_train, y_train)
print(knn_model.get_params())

y_pred = knn_model.predict(x_test)
accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))


#  MODEL TUNING
knn = KNeighborsClassifier()
np.arange(1,50)

# knn_params; aranacak olan parametre değerlerimiz
knn_params = {"n_neighbors": np.arange(1,50)}

knn_cv_model = GridSearchCV(knn, knn_params, cv = 10).fit(x_train, y_train)

knn_cv_model.best_score_

knn_cv_model.best_params_


# final modeli
knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(x_train, y_train)

# test setine ilişkin tahmin
y_pred = knn_tuned.predict(x_test)
accuracy_score(y_test, y_pred)

# accuracy yerine score kullanarak da aynı sonucu hesaplayabiliriz
knn_tuned.score(x_test, y_test)