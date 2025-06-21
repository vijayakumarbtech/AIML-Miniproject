# detect_image.py

import sys
import cv2
import joblib
import numpy as np
from scipy.stats import entropy
from skimage.filters import sobel

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

model = joblib.load("model.pkl")
image_path = sys.argv[1]

features = extract_features(image_path)
result = model.predict([features])[0]


print("✅ REAL IMAGE" if result == 0 else "🚫 FAKE IMAGE")
