import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Rastgele yeni veri seti oluşturulması
np.random.seed(0)
X = np.random.rand(100,1)*10
y = X*3 + np.random.randn(100,1)*2 +5
data = pd.DataFrame(np.hstack([X,y]), columns = ['X', 'y'])

# verilerin temizlenmesi 
data = data.dropna()  # null degerleri içeren satırları kaldırıyoruz
data = data[data.y > 0]  # y nin negatif yada sıfır oldugu satırları kaldırıyoruz

# veri setini test ve train olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['y'], test_size=0.25)

# liner regrasyon modeli olusturuyorum
lin_reg = LinearRegression()

# modeli dataya fit ediyoruz
lin_reg.fit(X_train, y_train)

# test daya üzerinde prediction yapıyoruz
lin_pred = lin_reg.predict(X_test)

# RMSE hesaplıyoruz test veri setinde
lin_rmse = sqrt(mean_squared_error(y_test, lin_pred))
print("Linear Regression RMSE: ", lin_rmse)

# RMSLE hesaplıyoruz test veri setinde yine
lin_rmsle = sqrt(mean_squared_log_error(y_test, lin_pred))
print("Linear Regression RMSLE: ", lin_rmsle)

#Karar agacı olusturuyoruz
dt_reg = DecisionTreeRegressor()

# modeli dataya eğitiyoruz
dt_reg.fit(X_train, y_train)

# test data üzerinde prediction yapıyoruz
dt_pred = dt_reg.predict(X_test)

# test data üzerinde RMSE hesaplıyoruz
dt_rmse = sqrt(mean_squared_error(y_test, dt_pred))
print("Decision Tree Regression RMSE: ", dt_rmse)

# RMSLE hesaplıyoruz test data üzerinde
dt_rmsle = sqrt(mean_squared_log_error(y_test, dt_pred))
print("Decision Tree Regression RMSLE: ", dt_rmsle)