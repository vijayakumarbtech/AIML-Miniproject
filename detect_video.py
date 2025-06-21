import cv2
import sys
import numpy as np
import joblib
from scipy.stats import entropy
from skimage.filters import sobel
from collections import Counter

def extract_features(image):
    img = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    edge_val = np.mean(sobel(gray))
    ent = entropy(hist)
    return np.hstack([hist, edge_val, ent])

def classify_video(video_path, model, frame_interval=30, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video: {video_path}")
    predictions = []
    frame_count = 0
    processed = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            try:
                features = extract_features(frame)
                pred = model.predict([features])[0]
                predictions.append(pred)
                processed += 1
            except Exception as e:
                print(f"⚠️ Skipping frame due to error: {e}")
        frame_count += 1

    cap.release()
    if not predictions:
        print("❌ No valid frames found.")
        return

    final = Counter(predictions).most_common(1)[0][0]
    print("✅ REAL VIDEO" if final == 0 else "🚫 FAKE VIDEO")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_video.py <video_path>")
        sys.exit(1)

    model = joblib.load("model.pkl")
    video_path = sys.argv[1]
    classify_video(video_path, model)
