import json
import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import shutil
from sklearn.metrics.pairwise import cosine_similarity

test_path = '/mnt/d/Fawry Comp/surveillance-for-retail-stores/face_identification/face_identification/ourtest2'
features_path = "./all_features.json"
targets_path = "./target.json"

def move_image(src_image_path, target):
    dest_path = os.path.join(test_path, target)
    os.makedirs(dest_path, exist_ok=True)  
    shutil.move(src_image_path, dest_path)
    #print(f"Moved {src_image_path} → {dest_path}")

with open(features_path, "r") as f:
    embeddings = np.array(json.load(f)) 

with open(targets_path, "r") as f:
    targets = json.load(f)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=224, margin=10, min_face_size=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

test_images = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

THRESHOLD = 0.5

fail_count = 0

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)

    if face is None:
        fail_count += 1
        print(f"No face detected in: {img_path}")
        continue

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        test_embeddings = facenet(face).cpu().numpy().flatten()
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings)      

    cos_similarities = cosine_similarity(embeddings, test_embeddings.reshape(1, -1))
    best_match_idx = np.argmax(cos_similarities)
    max_similarity = cos_similarities[best_match_idx][0]

    if max_similarity > THRESHOLD:
        predicted_person = "Unknown"
    else:
        predicted_person = targets[best_match_idx]
    move_image(img_path, predicted_person)

    print(f"Test Image: {img_path} -> Predicted Person: {predicted_person} (cos_similarity: {max_similarity:.4f})")

print(f"fail_count: {fail_count}")