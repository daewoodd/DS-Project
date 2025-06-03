import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_excel("./dataset/Rice_MSC_Dataset.xlsx")
X = df.drop(columns=["CLASS"])
y = df["CLASS"]

# Impute missing values (before normalization)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Save the imputer
joblib.dump(imputer, "models/imputer.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "models/scaler.pkl")

# Train models
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
svm = SVC(probability=True).fit(X_train_scaled, y_train)
nb = GaussianNB().fit(X_train_scaled, y_train)

# Save models
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(nb, "models/nb_model.pkl")

print("Training complete. Models saved.")

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluate and save results
X_test_scaled = scaler.transform(X_test)
results = {}

for name, model in [("knn", knn), ("svm", svm), ("nb", nb)]:
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    results[name] = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": model.classes_.tolist()
    }

# Save evaluation results
import json
with open("models/eval_results.json", "w") as f:
    json.dump(results, f)

# Save processed test data for PCA/t-SNE
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("models/X_test_scaled.csv", index=False)
y_test.to_csv("models/y_test.csv", index=False)

print("Evaluation results saved.")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate projections
projections = {}

# PCA projection
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_test_scaled)
projections["pca"] = [
    {"x": float(x), "y": float(y), "label": label}
    for (x, y), label in zip(pca_result, y_test)
]

# t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(X_test_scaled)
projections["tsne"] = [
    {"x": float(x), "y": float(y), "label": label}
    for (x, y), label in zip(tsne_result, y_test)
]

# Save both projections
with open("models/projections.json", "w") as f:
    json.dump(projections, f)

print("Projections saved.")
