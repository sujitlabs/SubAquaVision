"""
Configuration settings for Underwater Object Detection & Measurement System.
"""

# App Information
APP_TITLE = "SubAquaVisionUltra"
APP_SUBTITLE = "Underwater Object Detection, ROV Streaming & Measurement System"
APP_VERSION = "2.5.0"

# AI Model Settings
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-flash-latest"

# Object Category Color Scheme (Hex & RGB)
CATEGORY_CONFIG = {
    "marine_life": {
        "name": "Marine Life",
        "color_hex": "#00E676",  # Bright Green
        "color_rgb": (0, 230, 118),
        "icon": "??"
    },
    "debris": {
        "name": "Marine Debris / Trash",
        "color_hex": "#FF1744",  # High-vis Red
        "color_rgb": (255, 23, 68),
        "icon": "??"
    },
    "infrastructure": {
        "name": "Subsea Infrastructure",
        "color_hex": "#00E5FF",  # Electric Cyan
        "color_rgb": (0, 229, 255),
        "icon": "?"
    },
    "diver": {
        "name": "Diver / Human Activity",
        "color_hex": "#FFEA00",  # Vivid Yellow
        "color_rgb": (255, 234, 0),
        "icon": "??"
    },
    "other": {
        "name": "Other Unclassified",
        "color_hex": "#D500F9",  # Neon Purple
        "color_rgb": (213, 0, 249),
        "icon": "??"
    }
}

# ROV Camera Optical Defaults
DEFAULT_CAMERA_PARAMS = {
    "horizontal_fov_deg": 75.0,     # Standard ROV wide-angle lens (degrees)
    "default_sensor_width_mm": 6.17, # 1/2.3" sensor standard
    "default_focal_length_mm": 3.6,  # 3.6mm lens
    "reference_distance_m": 2.0,     # Default depth/distance baseline
    "calibration_scale_cm_per_px": 0.15 # Baseline at 1m distance
}

# Sample Stream URLs for Testing
SAMPLE_STREAMS = {
    "Local USB Camera": "0",
    "BlueROV2 Default RTSP": "rtsp://192.168.2.2:554/video",
    "ESP32-CAM Stream": "http://192.168.1.100:81/stream",
    "Generic MJPEG Stream": "http://192.168.1.50:8080/video"
}
