import cv2
import numpy as np
import json
import os
from typing import List, Dict


class ResultSaver:
    """Handles saving of detection results."""

    @staticmethod
    def save_results(
        image: np.ndarray, boxes: List[Dict[str, any]], output_dir: str, image_name: str
    ) -> None:
        """
        Save detection results as JSON and annotated image.

        Args:
            image (np.ndarray): Original image.
            boxes (List[Dict[str, any]]): List of bounding boxes with optional text.
            output_dir (str): Output directory.
            image_name (str): Base name for output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON
        json_path = os.path.join(output_dir, f"{image_name}_markup.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=4)

        # Draw bounding boxes
        for box in boxes:
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = box["label"]
            if box["label"] == "series_number":
                # Rotate text for series_number
                M = cv2.getRotationMatrix2D((x, y - 10), -90, 1.0)
                text_img = np.zeros_like(image)
                cv2.putText(
                    text_img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                text_img = cv2.warpAffine(text_img, M, (image.shape[1], image.shape[0]))
                image[text_img != 0] = text_img[text_img != 0]
            else:
                cv2.putText(
                    image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Save annotated image
        output_image_path = os.path.join(output_dir, f"{image_name}_annotated.png")
        cv2.imwrite(output_image_path, image)
        print(f"Results saved: {json_path}, {output_image_path}")
        