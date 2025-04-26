import os
import sys

from src.ImageProcessor import ImageProcessor
from src.ObjectDetector import ObjectDetector
from src.ResultSaver import ResultSaver


def main(image_path: str) -> None:
    """
    Main function to process the passport image.
    Args:
        image_path (str): Path to the input image.
    """
    processor = ImageProcessor(image_path)
    image, gray = processor.load_and_preprocess()

    detector = ObjectDetector()
    face_box = detector.detect_face(gray)
    text_boxes = detector.detect_text_areas(image)

    boxes = []
    if face_box:
        boxes.append(face_box)
    boxes.extend(text_boxes)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    ResultSaver.save_results(image, boxes, "output", image_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
    