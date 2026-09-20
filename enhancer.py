"""
Underwater Image & Video Frame Preprocessing & Enhancement Engine.
Implements Red-Channel Compensation, CLAHE, White Balance, and Dehazing.
"""

import cv2
import numpy as np
from PIL import Image

def compensate_red_channel(img_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Compensates for underwater red-channel attenuation using green and blue channels.
    """
    img_float = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    # Calculate difference between green/blue and red
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    # Compensate red channel where it has decayed significantly
    if mean_r < 1e-3:
        mean_r = 1e-3
    
    scale_factor = (mean_g + mean_b) / (2.0 * mean_r)
    scale_factor = np.clip(scale_factor, 1.0, 2.5)  # Avoid over-saturation
    
    r_comp = r * (1.0 + alpha * (scale_factor - 1.0))
    r_comp = np.clip(r_comp, 0, 255)
    
    compensated = cv2.merge([b, g, r_comp])
    return compensated.astype(np.uint8)

def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.5, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization in LAB color space.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def white_balance_simplest(img_bgr: np.ndarray, percent: int = 1) -> np.ndarray:
    """
    Simplest color balance algorithm to eliminate extreme color cast.
    """
    out_channels = []
    cum_percent = percent / 100.0 / 2.0
    
    for channel in cv2.split(img_bgr):
        low_val = np.percentile(channel, cum_percent * 100)
        high_val = np.percentile(channel, (1.0 - cum_percent) * 100)
        
        channel_scaled = np.clip(channel, low_val, high_val)
        if high_val > low_val:
            channel_scaled = ((channel_scaled - low_val) / (high_val - low_val) * 255.0).astype(np.uint8)
        else:
            channel_scaled = channel
        out_channels.append(channel_scaled)
        
    return cv2.merge(out_channels)

def enhance_underwater(image_input, method: str = "balanced") -> np.ndarray:
    """
    Complete underwater enhancement pipeline.
    
    Parameters:
        image_input: PIL.Image or np.ndarray (RGB or BGR)
        method: "fast", "balanced", or "deep_restoration"
        
    Returns:
        np.ndarray: Enhanced image in RGB format
    """
    if isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        # Assume RGB if 3 channels and standard array
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image_input
    else:
        raise ValueError("Unsupported image type")

    if method == "fast":
        # Fast CLAHE only
        enhanced_bgr = apply_clahe(img_bgr, clip_limit=2.0)
    elif method == "deep_restoration":
        # Full pipeline: Red comp + Simplest WB + Strong CLAHE + Unsharp Mask
        red_comp = compensate_red_channel(img_bgr, alpha=0.6)
        wb = white_balance_simplest(red_comp, percent=2)
        enhanced_bgr = apply_clahe(wb, clip_limit=3.0, grid_size=(8, 8))
        
        # Slight sharpening
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.3, gaussian, -0.3, 0)
    else:
        # Balanced (Default): Red comp + Balanced CLAHE + Color balance
        red_comp = compensate_red_channel(img_bgr, alpha=0.4)
        wb = white_balance_simplest(red_comp, percent=1)
        enhanced_bgr = apply_clahe(wb, clip_limit=2.2, grid_size=(8, 8))

    # Convert back to RGB for display & Gemini
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return enhanced_rgb
