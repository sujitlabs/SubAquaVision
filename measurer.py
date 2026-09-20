"""
Measurement & Optical Geometry Engine for Underwater Object Detection.
Estimates Real-World Physical Dimensions (cm/m) and Distance from ROV Camera.
"""

import math
from typing import Dict, Any, Tuple

# Refractive index of seawater (approx 1.333) causes ~33% optical magnification
WATER_REFRACTIVE_INDEX = 1.333

def calculate_focal_length_px(image_width_px: int, fov_deg: float = 75.0, in_water: bool = True) -> float:
    """
    Computes effective focal length in pixels using horizontal field of view.
    Accounts for water refraction flat-port lens effect if in_water=True.
    """
    fov_rad = math.radians(fov_deg)
    if in_water:
        # Apparent FOV narrows underwater due to Snell's law at flat port
        apparent_fov_rad = 2.0 * math.asin(math.sin(fov_rad / 2.0) / WATER_REFRACTIVE_INDEX)
    else:
        apparent_fov_rad = fov_rad
        
    focal_length_px = (image_width_px / 2.0) / math.tan(apparent_fov_rad / 2.0)
    return focal_length_px

def estimate_optical_distance(
    pixel_width: float, 
    real_width_cm: float, 
    image_width_px: int, 
    fov_deg: float = 75.0
) -> float:
    """
    Estimates distance (meters) to the object using pinhole geometry:
    D = (f_px * Real_Width) / Pixel_Width
    """
    if pixel_width <= 0:
        return 0.0
    f_px = calculate_focal_length_px(image_width_px, fov_deg, in_water=True)
    distance_cm = (f_px * real_width_cm) / pixel_width
    distance_m = distance_cm / 100.0
    return round(distance_m, 2)

def estimate_object_dimensions(
    box_normalized: list, 
    image_width_px: int, 
    image_height_px: int, 
    distance_m: float, 
    fov_deg: float = 75.0
) -> Dict[str, float]:
    """
    Given normalized box [ymin, xmin, ymax, xmax] (0 to 1000 or 0 to 1) and estimated distance,
    calculates real-world Width and Height in centimeters.
    """
    ymin, xmin, ymax, xmax = box_normalized
    # Normalize to 0.0 - 1.0 if passed as 0 - 1000
    if ymax > 1.0 or xmax > 1.0:
        ymin, xmin, ymax, xmax = ymin/1000.0, xmin/1000.0, ymax/1000.0, xmax/1000.0
        
    box_w_px = (xmax - xmin) * image_width_px
    box_h_px = (ymax - ymin) * image_height_px
    
    f_px = calculate_focal_length_px(image_width_px, fov_deg, in_water=True)
    
    distance_cm = distance_m * 100.0
    real_width_cm = (box_w_px * distance_cm) / f_px
    real_height_cm = (box_h_px * distance_cm) / f_px
    
    return {
        "width_cm": round(real_width_cm, 1),
        "height_cm": round(real_height_cm, 1),
        "diagonal_cm": round(math.sqrt(real_width_cm**2 + real_height_cm**2), 1),
        "pixel_width": round(box_w_px, 1),
        "pixel_height": round(box_h_px, 1)
    }

def format_measurement_string(dims: Dict[str, Any], distance_m: float) -> str:
    """
    Formats size and distance in simple, readable marine notation.
    Example: 'Size: 34.5 ? 12.0 cm | Dist: 1.8 m'
    """
    w = dims.get("width_cm", 0.0)
    h = dims.get("height_cm", 0.0)
    if w >= 100 or h >= 100:
        # Format in meters if large
        return f"Size: {w/100:.2f}m ? {h/100:.2f}m | Dist: {distance_m:.1f}m"
    return f"Size: {w:.1f} ? {h:.1f} cm | Dist: {distance_m:.1f}m"
