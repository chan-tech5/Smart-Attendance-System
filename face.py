import cv2
import os
import numpy as np
from datetime import datetime

# Initialize the face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the dataset folder
datasets = 'datasets'  # This folder should contain subfolders named after persons

print('Training the model...')

# Prepare training data
(images, labels, names, id) = ([], [], {}, 0)

# Walk through the dataset folder to collect names and images
for (subdirs, dirs, _) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Folder name is the person's name
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id  # The ID is based on the folder index (this will be ignored later)
            images.append(cv2.imread(path, 0))  # Load in grayscale
            labels.append(int(label))
        id += 1

# Convert images and labels into NumPy arrays
(images, labels) = [np.array(lst) for lst in [images, labels]]

# Train the model on the collected dataset
model.train(images, labels)

# Attendance logging function
def mark_attendance(name):
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        data = f.readlines()
        attendance_list = [line.split(',')[0] for line in data]

        # If the person is not already logged, add them
        if name not in attendance_list:
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{timestamp}\n')
            print(f"Attendance marked for {name}")

# Start the video capture for real-time face recognition
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (130, 100))

        # Predict the face
        prediction = model.predict(face_resized)
        confidence = int(100 * (1 - (prediction[1] / 300)))  # Confidence score

        if confidence > 50:  # Only accept predictions with high confidence
            name = names[prediction[0]]  # Get the name from the folder
            cv2.putText(frame, f'{name} ({confidence}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mark_attendance(name)  # Log attendance with the name
        else:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the video feed with face recognition
    cv2.imshow('Face Recognition Attendance', frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# Display attendance data (names dictionary)
print("Attendance dataset (names):", names)
