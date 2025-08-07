import joblib


knn_wine_model=joblib.load( "knn_wine_model.pkl")
example = [[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]]
result = knn_wine_model.predict(example)
print("Prediction:", result)
