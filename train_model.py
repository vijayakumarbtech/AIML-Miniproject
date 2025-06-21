# train_model.py

import os
import cv2
import numpy as np
import joblib
from scipy.stats import entropy
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    edge_val = np.mean(sobel(gray))
    ent = entropy(hist)

    features = np.hstack([hist, edge_val, ent])
    return features

X, y = [], []

# Real images = 0
for file in os.listdir("datasets/real"):
    path = os.path.join("datasets/real", file)
    if file.endswith(".jpg") or file.endswith(".png"):
        X.append(extract_features(path))
        y.append(0)

# Fake images = 1
for file in os.listdir("datasets/fake"):
    path = os.path.join("datasets/fake", file)
    if file.endswith(".jpg") or file.endswith(".png"):
        X.append(extract_features(path))
        y.append(1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
