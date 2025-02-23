from insightface.app import FaceAnalysis
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img = cv2.imread("ourDataset/train/Akshay Kumar/Akshay Kumar_0.jpg")
h,w, _ = img.shape

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(w, h))



#print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = app.get(img)
print(faces)
print(faces[0].embedding)
# if faces:
#     face_embedding = faces[0].embedding  # 512-D vector
#     print(face_embedding)  # You can save this in a database for later recognition
