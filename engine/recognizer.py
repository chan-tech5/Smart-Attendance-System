import cv2
import os
import numpy as np
import pickle


class FaceRecognizer:
    def __init__(self, model_type='FisherFace', datasets_dir='datasets'):
        self.model_type  = model_type
        self.datasets_dir = datasets_dir
        self.names        = {}
        self.fisher_model = None
        self.lbph_model   = None
        self._load()

    def _load(self):
        # ── Always load names first — works for both FisherFace AND LBPH ──
        if os.path.exists('names.pkl'):
            try:
                with open('names.pkl', 'rb') as f:
                    self.names = pickle.load(f)
                print(f"[Recognizer] Names loaded: {self.names}")
            except Exception as e:
                print(f"[Recognizer] Names load error: {e}")

        # ── Try FisherFace ──────────────────────────────────────────────
        self.fisher_model = None
        if os.path.exists('fisher_model.yml'):
            try:
                m = cv2.face.FisherFaceRecognizer_create()
                m.read('fisher_model.yml')
                self.fisher_model = m
                print("[Recognizer] FisherFace model loaded.")
            except Exception as e:
                print(f"[Recognizer] Fisher load error: {e}")

        # ── Try LBPH ────────────────────────────────────────────────────
        self.lbph_model = None
        if os.path.exists('trained_model.yml'):
            try:
                m = cv2.face.LBPHFaceRecognizer_create()
                m.read('trained_model.yml')
                self.lbph_model = m
                print("[Recognizer] LBPH model loaded.")
            except Exception as e:
                print(f"[Recognizer] LBPH load error: {e}")

    def recognize(self, frame, bbox):
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            return "Unknown", 0
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (130, 100))

        # ── FisherFace (primary — best for 2+ people) ───────────────────
        if self.fisher_model and self.names:
            try:
                label, conf = self.fisher_model.predict(gray)
                score = max(0, int(100 - conf / 10))
                if score > 40:
                    return self.names.get(label, "Unknown"), score
            except Exception:
                pass

        # ── LBPH (fallback — works with 1 person too) ───────────────────
        if self.lbph_model and self.names:
            try:
                label, conf = self.lbph_model.predict(gray)
                # LBPH: lower conf = better match (0 is perfect, ~100 is poor)
                if conf < 80:
                    score = max(10, int(100 - conf))
                    name  = self.names.get(label, "Unknown")
                    return name, score
            except Exception:
                pass

        return "Unknown", 0

    def reload(self):
        self.names        = {}
        self.fisher_model = None
        self.lbph_model   = None
        self._load()
