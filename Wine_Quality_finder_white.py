import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.neighbors as ng
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mydata=pd.read_csv("winequality_white.csv")
x=mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=mydata["quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn_wine_model_white= Sequential()
knn_wine_model_white.add(Dense(10, activation="relu", input_shape=(4,)))
knn_wine_model_white.add(Dense(10, activation="relu"))
knn_wine_model_white.add(Dense(10, activation="relu"))
knn_wine_model_white.add(Dense(1))
knn_wine_model_white.compile(optimizer="adam", loss="mean_squared_error")

knn_wine_model_white=ng.KNeighborsClassifier(n_neighbors=3)
knn_wine_model_white.fit(x_train,y_train)
print("Training has completed...")
test_result=knn_wine_model_white.predict(x_test)
print("RMSE",np.sqrt(mean_squared_error(y_test,test_result)))
knn_wine_model_white=joblib.dump(knn_wine_model_white,"knn_wine_model_white.pkl")
