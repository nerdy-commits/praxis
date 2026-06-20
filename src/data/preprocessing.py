# =============================================================================
#  Image Preprocessing — Ben Graham Method
# =============================================================================
"""
Retinal fundus image preprocessing following Ben Graham's approach
from the 2015 Kaggle Diabetic Retinopathy competition.

The key insight: subtract a Gaussian-blurred version of the image from itself
to enhance vessel and lesion contrast while removing uneven illumination.

Reference: https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801
"""

import cv2
import numpy as np


class BenGrahamPreprocessor:
    """
    Ben Graham's preprocessing pipeline for retinal fundus images.

    Steps:
        1. Crop the black border around the circular fundus region
        2. Resize to target dimensions
        3. Subtract a Gaussian-blurred version (background removal)
        4. Clip and rescale to [0, 255]

    Parameters
    ----------
    image_size : int
        Target square dimension for output images.
    sigma : int
        Gaussian blur kernel sigma. Controls how much background to subtract.
        Higher values = more aggressive background removal.
    """

    def __init__(self, image_size: int = 224, sigma: int = 10):
        self.image_size = image_size
        self.sigma = sigma

    def crop_to_circle(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-crop the black border around the circular fundus region.
        Finds the bounding box of non-black pixels and crops to it.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Threshold to find the fundus region
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours, pick the largest
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = image[y : y + h, x : x + w]
        return cropped

    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        """
        Subtract Gaussian-blurred background from the image.
        Formula: output = image - gaussian_blur(image) + 128

        The +128 shifts the result to a visible range since
        subtraction can produce negative values.
        """
        kernel_size = 0  # auto-computed from sigma by OpenCV
        blurred = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), self.sigma
        )
        # Compute in float to avoid uint8 underflow
        result = image.astype(np.float32) - blurred.astype(np.float32) + 128.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the full Ben Graham preprocessing pipeline."""
        image = self.crop_to_circle(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.subtract_background(image)
        return image


def preprocess_image(
    image: np.ndarray,
    image_size: int = 224,
    sigma: int = 10,
) -> np.ndarray:
    """
    Functional interface to Ben Graham preprocessing.

    Parameters
    ----------
    image : np.ndarray
        RGB image as HxWx3 numpy array.
    image_size : int
        Target output size (square).
    sigma : int
        Gaussian blur sigma for background subtraction.

    Returns
    -------
    np.ndarray
        Preprocessed image of shape (image_size, image_size, 3).
    """
    preprocessor = BenGrahamPreprocessor(image_size=image_size, sigma=sigma)
    return preprocessor(image)
