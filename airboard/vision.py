import cv2
import mediapipe as mp
import numpy as np
from .models import paths
from .core import IdentityGate, classify


class FaceMatcher:
    """Single-presenter convenience filter, NOT authentication or liveness."""

    def __init__(self, reference, threshold=0.45):
        detector, recognizer = paths()
        self.detector = cv2.FaceDetectorYN.create(
            str(detector), "", (320, 320), 0.85, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.threshold = threshold
        self.gate = IdentityGate()
        image = cv2.imread(str(reference))
        if image is None:
            raise ValueError(
                f"Не удалось прочитать фото: {reference}. Добавь person.jpg с одним лицом."
            )
        image = self.resize(image, 1000)
        faces = self.detect(image)
        if len(faces) != 1:
            raise ValueError(
                f"На эталонном фото должно быть ровно одно лицо; найдено: {len(faces)}"
            )
        self.reference = self.feature(image, faces[0])

    @staticmethod
    def resize(image, width):
        h, w = image.shape[:2]
        return cv2.resize(image, (width, round(h * width / w))) if w > width else image

    def detect(self, image):
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return [] if faces is None else faces

    def feature(self, image, face):
        return self.recognizer.feature(self.recognizer.alignCrop(image, face)).copy()

    def check(self, frame):
        # Re-evaluate every frame: no stale authorization after a person leaves.
        image = self.resize(frame, 640)
        faces = self.detect(image)
        if len(faces) != 1:
            self.gate.update(False)
            return (
                False,
                "ONE PERSON ONLY" if len(faces) > 1 else "LOOK AT CAMERA",
                None,
            )
        score = float(
            self.recognizer.match(
                self.reference,
                self.feature(image, faces[0]),
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )
        accepted = self.gate.update(np.isfinite(score) and score >= self.threshold)
        return (
            accepted,
            "READY"
            if accepted
            else "VERIFYING"
            if score >= self.threshold
            else "PERSON NOT RECOGNIZED",
            score,
        )


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def check(self, mirrored_frame):
        rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        found = result.multi_hand_landmarks or []
        if len(found) != 1:
            return "idle", None, len(found)
        h, w = mirrored_frame.shape[:2]
        points = [(p.x * w, p.y * h) for p in found[0].landmark]
        return classify(points), points, 1

    def close(self):
        self.hands.close()
