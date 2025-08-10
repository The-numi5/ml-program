import joblib


knn_wine_model_white=joblib.load( "knn_wine_model_white.pkl")
example = [[6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0]]
result = knn_wine_model_white.predict(example)
print("Prediction:", result)