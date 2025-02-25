import json
import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

base_path = "/mnt/d/Fawry Comp/surveillance-for-retail-stores/face_identification/face_identification/train"
save_dir = "./"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=224, margin=10, min_face_size=40, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
targets = []

fail_count = 0

person_dirs = [os.path.join(base_path, person) for person in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, person))]

for person_dir in person_dirs:
    person_name = os.path.basename(person_dir)  
    image_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_path in image_paths:
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
            train_embeddings = facenet(face).cpu().numpy().flatten()

        train_embeddings = train_embeddings / np.linalg.norm(train_embeddings)  
        #print(len(train_embeddings))
        embeddings.append(train_embeddings.tolist())
        print(person_name)
        targets.append(person_name)

print(f"fail count: {fail_count}")

with open(os.path.join(save_dir, "all_features.json"), "w") as f:
    json.dump(embeddings, f, indent=4)
print(f"Saved embeddings in all_features.json")

with open(os.path.join(save_dir, "target.json"), "w") as f:
    json.dump(targets, f, indent=4)
print(f"Saved targets in target.json")
