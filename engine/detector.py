import cv2
import os

class FaceDetector:
    def __init__(self):
        cascade_path = 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
        return faces if len(faces) > 0 else []
