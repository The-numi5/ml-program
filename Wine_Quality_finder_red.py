import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.neighbors as ng
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mydata=pd.read_csv("winequality_red.csv")
x=mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=mydata["quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn_wine_model_red= Sequential()
knn_wine_model_red.add(Dense(10, activation="relu", input_shape=(4,)))
knn_wine_model_red.add(Dense(10, activation="relu"))
knn_wine_model_red.add(Dense(10, activation="relu"))
knn_wine_model_red.add(Dense(1))
knn_wine_model_red.compile(optimizer="adam", loss="mean_squared_error")

knn_wine_model_red=ng.KNeighborsClassifier(n_neighbors=3)
knn_wine_model_red.fit(x_train,y_train)
print("Training has completed...")
test_result=knn_wine_model_red.predict(x_test)
print("RMSE",np.sqrt(mean_squared_error(y_test,test_result)))
knn_wine_model_red=joblib.dump(knn_wine_model_red,"knn_wine_model_red.pkl")
