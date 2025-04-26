import cv2
import numpy as np
import os
import requests
from typing import List, Dict, Optional


class ObjectDetector:
    """Handles detection of faces and text areas in the image."""

    MODEL_URL = (
        "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    )
    MODEL_PATH = "models/frozen_east_text_detection.pb"

    def __init__(self) -> None:
        """Initialize the ObjectDetector with required models."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._ensure_model_exists()
        self.text_net = cv2.dnn.readNet(self.MODEL_PATH)

    def _ensure_model_exists(self) -> None:
        """Ensure the EAST model exists, download if necessary."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"Model not found at {self.MODEL_PATH}. Downloading...")
            self._download_model()
        else:
            print(f"Model found at {self.MODEL_PATH}.")

    def _download_model(self) -> None:
        """Download the EAST model from the official repository."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(self.MODEL_URL, stream=True)
            response.raise_for_status()
            with open(self.MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {self.MODEL_PATH}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def detect_face(self, gray: np.ndarray) -> Optional[Dict[str, int]]:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return {"label": "photo", "x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        return None

    def detect_text_areas(self, image: np.ndarray) -> List[Dict[str, int]]:
        """
        Detect text areas using EAST text detector and assign passport-specific labels.
        """
        orig_h, orig_w = image.shape[:2]
        # Detect face to define left margin
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = self.detect_face(gray)
        left_margin = face['x'] + face['width'] if face else orig_w * 0.2

        # Prepare image for EAST
        new_w = ((orig_w + 31) // 32) * 32
        new_h = ((orig_h + 31) // 32) * 32
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (new_w, new_h)),
            1.0,
            (new_w, new_h),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self.text_net.setInput(blob)
        scores, geometry = self.text_net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])

        # Decode predictions
        boxes, confidences = [], []
        conf_thresh = 0.5
        rows, cols = scores.shape[2:4]
        for y in range(rows):
            for x in range(cols):
                score = scores[0, 0, y, x]
                if score < conf_thresh:
                    continue
                # Geometry
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = geometry[0, 4, y, x]
                cos, sin = np.cos(angle), np.sin(angle)
                h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
                w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
                end_x = int(offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
                end_y = int(offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                # Scale back
                start_x = int(start_x / scale_x)
                start_y = int(start_y / scale_y)
                w = int(w / scale_x)
                h = int(h / scale_y)
                # Exclude face region
                if start_x < left_margin:
                    continue
                boxes.append([start_x, start_y, w, h])
                confidences.append(float(score))

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()
        filtered = [boxes[i] for i in indices]

        # Separate rotated series/number boxes
        series_boxes = []
        text_boxes = []
        right_edge = orig_w * 0.9
        for b in filtered:
            x, y, w, h = b
            # highly rotated and near right edge
            if x > right_edge and h > w * 1.5:
                series_boxes.append(b)
            else:
                text_boxes.append(b)

        # Label rows
        # Cluster by y coordinate
        text_boxes.sort(key=lambda b: b[1])
        rows = []  # list of lists
        y_tol = 20
        for b in text_boxes:
            placed = False
            for row in rows:
                if abs(b[1] - row[0][1]) < y_tol:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        # Expected rows: 5 (surname, name, patronymic, gender/date, place)
        labels = ['surname', 'name', 'patronymic', 'gender_date', 'birth_place']
        results = []
        for i, row in enumerate(rows[:5]):
            row.sort(key=lambda b: b[0])  # by x
            if labels[i] == 'gender_date':
                # two entries: gender then date
                if len(row) >= 2:
                    lib = ['gender', 'birth_date']
                    for j, b in enumerate(row[:2]):
                        x, y, w, h = b
                        results.append({'label': lib[j], 'x': x, 'y': y, 'width': w, 'height': h})
                else:
                    for j, key in enumerate(['gender', 'birth_date']):
                        if j < len(row):
                            x, y, w, h = row[j]
                            results.append({'label': key, 'x': x, 'y': y, 'width': w, 'height': h})
            else:
                key = labels[i]
                # if multiple boxes (unlikely), take first
                x, y, w, h = row[0]
                results.append({'label': key, 'x': x, 'y': y, 'width': w, 'height': h})

        # Series and number: take rightmost
        if series_boxes:
            b = max(series_boxes, key=lambda bb: bb[0])
            x, y, w, h = b
            results.append({'label': 'series_number', 'x': x, 'y': y, 'width': w, 'height': h})

        return results
