import joblib

knn = joblib.load("models/knn_model.pkl")
svm = joblib.load("models/svm_model.pkl")
nb = joblib.load("models/nb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

MODELS = {
    "knn": knn,
    "svm": svm,
    "nb": nb
}