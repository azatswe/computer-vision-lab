import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from airboard.models import paths, verified
from airboard.session import Session
from airboard.ui import render, export
from airboard.vision import FaceMatcher, HandTracker
import app


class IntegrationTests(unittest.TestCase):
    def test_face_models_load_and_blank_face_rejected(self):
        detector, recognizer = paths()
        net = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), 0.85)
        _, faces = net.detect(np.zeros((320, 320, 3), np.uint8))
        self.assertIsNone(faces)
        recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        features = recognizer.feature(np.full((112, 112, 3), 100, np.uint8))
        self.assertEqual(features.shape, (1, 128))
        self.assertTrue(np.isfinite(features).all())

    def test_bad_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "blank.png"
            cv2.imwrite(str(file), np.zeros((320, 320, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "ровно одно лицо"):
                FaceMatcher(file)

    def test_real_hand_inference_blank_frame(self):
        tracker = HandTracker()
        try:
            gesture, points, count = tracker.check(np.zeros((360, 640, 3), np.uint8))
            self.assertEqual((gesture, points, count), ("idle", None, 0))
        finally:
            tracker.close()

    def test_preview_shape_and_export_no_camera(self):
        session = Session()
        session.board = True
        camera = np.full((480, 640, 3), (0, 0, 255), np.uint8)
        session.drawing.add((0.5, 0.5))
        image = render(session, camera, True, "READY", "draw", 0.7, 25)
        self.assertEqual(image.shape, (800, 1280, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = export(session.drawing, folder)
            saved = cv2.imread(str(path))
            self.assertEqual(saved.shape, (900, 1600, 3))
            self.assertFalse(np.any(np.all(saved == (0, 0, 255), axis=2)))
            self.assertNotEqual(path, export(session.drawing, folder))

    def test_corrupted_model_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad"
            path.write_bytes(b"broken")
            self.assertFalse(verified(path, "0" * 64))

    def test_invalid_cli(self):
        with self.assertRaises(ValueError):
            app.main(["--threshold", "nan"])
        with self.assertRaises(ValueError):
            app.main(["--camera", "-1"])

    def test_missing_photo_does_not_open_camera(self):
        with patch("cv2.VideoCapture") as capture:
            with self.assertRaises(ValueError):
                app.main(["--person", "/nonexistent/airboard-photo.jpg"])
            capture.assert_not_called()

    def test_camera_failure_releases_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / "person.jpg"
            photo.touch()
            with (
                patch("airboard.vision.FaceMatcher"),
                patch("airboard.vision.HandTracker") as hand,
                patch("cv2.VideoCapture") as camera,
                patch("cv2.destroyAllWindows"),
            ):
                camera.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(RuntimeError, "Камера недоступна"):
                    app.main(["--person", str(photo)])
                camera.return_value.release.assert_called_once()
                hand.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
