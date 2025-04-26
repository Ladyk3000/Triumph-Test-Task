import cv2
import numpy as np
from typing import Optional, Tuple


class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, image_path: str) -> None:
        """
        Initialize the ImageProcessor with the path to the image.

        Args:
            image_path (str): Path збор
            image_path (str): Path to the input image.
        """
        self.image_path: str = image_path
        self.image: Optional[np.ndarray] = None
        self.gray: Optional[np.ndarray] = None

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Original image and grayscale image.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Failed to load image")

        # Convert to grayscale and apply Gaussian blur
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
        return self.image, self.gray
    