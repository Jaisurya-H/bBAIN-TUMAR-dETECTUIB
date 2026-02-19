import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Defines the dataset path - User must popuplate this!
DATASET_PATH = "dataset"  # Folder containing 4 class folders: glioma, meningioma, notumor, pituitary

X = []
y = []

labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Check if dataset exists before proceeding
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset directory '{DATASET_PATH}' not found.")
    print("Please create the 'dataset' folder and add the 'glioma', 'meningioma', 'notumor', and 'pituitary' subfolders with images.")
    exit(1)

print("Loading images...")

for label_index, label in enumerate(labels):
    folder_path = os.path.join(DATASET_PATH, label)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist. Skipping.")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (200, 200))
            img = cv2.medianBlur(img, 5)

            img = img.flatten()

            X.append(img)
            y.append(label_index)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if not X:
    print("No images found. Please check your dataset.")
    exit(1)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} images. Splitting and training...")

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully! 'model.pkl' and 'scaler.pkl' are ready.")
