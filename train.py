import cv2
import os
import numpy as np

# Initialize face recognizer and face detector
model = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
datasets = 'datasets'

print('Training the model...')

# Prepare training data
(images, labels, ids, id) = ([], [], {}, 0)

for (subdirs, dirs, _) in os.walk(datasets):
    for subdir in dirs:
        ids[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))  # Load in grayscale
            labels.append(int(label))
        id += 1

# Convert data to NumPy arrays
(images, labels) = [np.array(lst) for lst in [images, labels]]

# Train the model
model.train(images, labels)

# Save the trained model
model.save('trained_model.yml')
print("Model trained and saved as 'trained_model.yml'.")
