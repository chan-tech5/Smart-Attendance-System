import cv2
import os
import numpy as np
import pickle


class FaceTrainer:
    def __init__(self, datasets_dir='datasets'):
        self.datasets_dir = datasets_dir

    def train(self):
        if not os.path.exists(self.datasets_dir):
            return False, "No datasets folder found. Please register users first."

        images, labels, names, id_ = [], [], {}, 0
        for _, dirs, _ in os.walk(self.datasets_dir):
            for subdir in sorted(dirs):
                subject_path = os.path.join(self.datasets_dir, subdir)
                imgs_found = 0
                for filename in os.listdir(subject_path):
                    img = cv2.imread(os.path.join(subject_path, filename), 0)
                    if img is not None:
                        images.append(cv2.resize(img, (130, 100)))
                        labels.append(id_)
                        imgs_found += 1
                if imgs_found > 0:
                    names[id_] = subdir
                    id_ += 1
            break  # only top-level dirs

        num_subjects = len(set(labels))

        if num_subjects == 0 or len(images) == 0:
            return False, "No face images found. Please enroll at least one person first."

        # ── FisherFace requires ≥2 subjects — fall back to LBPH for 1 person ──
        if num_subjects >= 2:
            try:
                model = cv2.face.FisherFaceRecognizer_create()
                model.train(np.array(images), np.array(labels))
                model.save('fisher_model.yml')
                with open('names.pkl', 'wb') as f:
                    pickle.dump(names, f)
                return True, f"FisherFace model trained — {len(images)} images, {num_subjects} subjects."
            except Exception as e:
                return False, f"FisherFace training error: {e}"
        else:
            # Single subject — use LBPH
            try:
                model = cv2.face.LBPHFaceRecognizer_create()
                model.train(np.array(images), np.array(labels))
                model.save('trained_model.yml')
                with open('names.pkl', 'wb') as f:
                    pickle.dump(names, f)
                return True, (
                    f"LBPH model trained — {len(images)} images for '{list(names.values())[0]}'. "
                    "Enroll a second person and re-train to upgrade to FisherFace."
                )
            except Exception as e:
                return False, f"LBPH training error: {e}"
