import json
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

base_path = "/mnt/d/Fawry Comp/surveillance-for-retail-stores/face_identification/face_identification/train"

w, h = 256, 256
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(w, h))

person_dirs = [
    os.path.join(base_path, person)
    for person in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, person))
]

for person_dir in person_dirs:
    person_name = os.path.basename(person_dir)
    embeddings = []

    image_paths = [
        os.path.join(person_dir, img)
        for img in os.listdir(person_dir)
        if img.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_path in image_paths:
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

        embeddings.append(face[0].normed_embedding.tolist())

    json_filename = os.path.join('Fawry-Competition/Face_Recognition/features', f"{person_name}.json")
    with open(json_filename, "w") as f:
        json.dump(embeddings, f, indent=4)

    print(f"Saved embeddings for {person_name} in {json_filename}")