import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import shutil

test_path = '/mnt/d/Fawry Comp/surveillance-for-retail-stores/face_identification/face_identification/ourtest'
features_path = "Fawry-Competition/Face_Recognition/all_features.json"
targets_path = "Fawry-Competition/Face_Recognition/target.json"

def move_image(src_image_path, target):
    dest_path = os.path.join(test_path, "person_" + target)
    os.makedirs(dest_path, exist_ok=True)  
    shutil.move(src_image_path, dest_path)
    print(f"Moved {src_image_path} → {dest_path}")

with open(features_path, "r") as f:
    embeddings = np.array(json.load(f)) 

with open(targets_path, "r") as f:
    targets = json.load(f)  

w, h = 128, 128
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(w, h))

test_images = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

THRESHOLD = 1.0

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = app.get(img)
    if len(face) == 0:
        print(f"No face detected in: {img_path}")
        continue
    
    test_embedding = np.array(face[0].normed_embedding)  
    distances = np.linalg.norm(embeddings - test_embedding, axis=1)  
    best_match_idx = np.argmin(distances)
    min_distance = distances[best_match_idx]
    if min_distance > THRESHOLD:
        predicted_person = "Unknown"
    else:
        predicted_person = targets[best_match_idx]
    move_image(img_path, predicted_person)

    print(f"Test Image: {img_path} -> Predicted Person: {predicted_person} (Distance: {distances[best_match_idx]:.4f})")
