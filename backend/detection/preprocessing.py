import cv2
import numpy as np


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


def _get_dark_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Computes the dark channel of an image.

    Args:
        image: The input image as a numpy array.
        window_size: The size of the local patch for minimum filter.

    Returns:
        The computed dark channel.
    """
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel


def _get_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """
    Estimates the atmospheric light of an image based on the dark channel.

    Args:
        image: The input image.
        dark_channel: The corresponding dark channel.
        top_percent: The percentage of brightest pixels in the dark channel to consider.

    Returns:
        The estimated atmospheric light vector (3,).
    """
    height, width = image.shape[:2]
    num_pixels = height * width
    num_top = max(int(num_pixels * top_percent), 1)

    dark_vec = dark_channel.reshape(-1)
    img_vec = image.reshape(-1, 3)

    indices = np.argsort(dark_vec)[::-1]
    top_indices = indices[:num_top]

    # Calculate average atmospheric light from the brightest pixels in the dark channel
    atmospheric_light = np.mean(img_vec[top_indices], axis=0)
    return atmospheric_light


def _get_transmission(image: np.ndarray, atmospheric_light: np.ndarray, window_size: int = 15, omega: float = 0.95) -> np.ndarray:
    """
    Estimates the transmission map using the dark channel prior.

    Args:
        image: The input image.
        atmospheric_light: The estimated atmospheric light.
        window_size: The size of the local patch.
        omega: Constant controlling the amount of haze removed.

    Returns:
        The estimated transmission map.
    """
    norm_img = image / atmospheric_light
    norm_dark = _get_dark_channel(norm_img, window_size)
    transmission = 1.0 - omega * norm_dark
    return transmission


def _guided_filter(img: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Applies a guided filter to refine the transmission map.

    Args:
        img: The guidance image (grayscale normalized to [0, 1]).
        p: The filtering input image (transmission map).
        radius: The local window radius.
        eps: Regularization parameter.

    Returns:
        The refined transmission map.
    """
    mean_i = cv2.boxFilter(img, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_ip = cv2.boxFilter(img * p, cv2.CV_64F, (radius, radius))
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(img * img, cv2.CV_64F, (radius, radius))
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    q = mean_a * img + mean_b
    return q


def apply_dehazing(frame: np.ndarray) -> np.ndarray:
    """
    Applies dehazing to an image using the Dark Channel Prior algorithm.

    Args:
        frame: The input BGR image.

    Returns:
        The dehazed BGR image.
    """
    i_float = frame.astype(np.float64)
    
    # 1. Estimate Dark Channel
    dark = _get_dark_channel(i_float, window_size=15)
    
    # 2. Estimate Atmospheric Light
    atmospheric_light = _get_atmospheric_light(i_float, dark)
    
    # 3. Estimate Transmission Map
    te = _get_transmission(i_float, atmospheric_light, window_size=15)
    
    # 4. Refine Transmission Map using Guided Filter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_norm = np.float64(gray) / 255.0
    t_refined = _guided_filter(gray_norm, te, radius=60, eps=0.0001)
    
    # 5. Recover the Scene Radiance
    t0 = 0.1
    t = np.maximum(t_refined, t0)
    t = cv2.merge([t, t, t])
    
    recovered = (i_float - atmospheric_light) / t + atmospheric_light
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    
    return recovered


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for low-light enhancement.

    Args:
        frame: The input BGR image.

    Returns:
        The CLAHE-enhanced BGR image.
    """
    # Convert image to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L (Lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocesses a video frame using Denoising, Dehazing (Dark Channel Prior), and CLAHE.

    Args:
        frame (np.ndarray): The input video frame (BGR format) to be preprocessed.

    Returns:
        np.ndarray: The preprocessed frame.

    Raises:
        PreprocessingError: If the input frame is None or not a valid numpy array.
    """
    if frame is None:
        raise PreprocessingError("Input frame is None. Cannot preprocess.")
    if not isinstance(frame, np.ndarray):
        raise PreprocessingError(f"Input frame must be a numpy.ndarray, got {type(frame)}.")
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise PreprocessingError("Input frame must be a 3-channel BGR image.")

    try:
        # Step 1: Denoising
        # Using fastNlMeansDenoisingColored for robust noise removal while preserving edges.
        # h=10, hColor=10, templateWindowSize=7, searchWindowSize=21 are standard parameters.
        denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # Step 2: Dehazing using Dark Channel Prior
        dehazed_frame = apply_dehazing(denoised_frame)

        # Step 3: Low-light enhancement using CLAHE
        final_frame = apply_clahe(dehazed_frame)

        return final_frame

    except Exception as e:
        raise PreprocessingError(f"An error occurred during frame preprocessing: {str(e)}")
