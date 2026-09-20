"""
Gemini Multimodal Vision Engine for Underwater Object Detection & Environmental Analysis.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from config import CATEGORY_CONFIG, DEFAULT_GEMINI_MODEL
from measurer import estimate_object_dimensions, format_measurement_string

# Structured system prompt for Gemini Flash
DETECTION_SYSTEM_PROMPT = """
You are an expert underwater computer vision and marine science AI.
Your mission is to perform high-accuracy object detection, physical measurement estimation, and environmental analysis on underwater ROV/AUV camera imagery.

Analyze the provided camera frame carefully. Detect ALL visible objects belonging to these categories:
1. "marine_life" (fish, sea turtles, corals, aquatic fauna and marine organisms)
2. "debris" (plastic bottles, ghost nets, cans, tires, plastic waste, discarded objects)
3. "infrastructure" (subsea pipelines, cables, shipwrecks, metal/concrete structures, equipment)
4. "diver" (human beings, ROV operators, scuba divers, swimmers, human faces)
5. "other" (smartphones, electronic devices, tools, furniture, room objects, geological formations)

For EVERY detected object, output a JSON object with:
- "box_2d": [ymin, xmin, ymax, xmax] normalized on a scale of 0 to 1000 (integers).
- "label": Clear common name (e.g. "Tiger Shark", "Plastic Bottle", "Subsea Flange", "Scuba Diver").
- "scientific_name": Scientific/taxonomic name or technical descriptor (e.g. "Galeocerdo cuvier", "Polyethylene Terephthalate").
- "category": One of ["marine_life", "debris", "infrastructure", "diver", "other"].
- "confidence": Float between 0.50 and 0.99 indicating detection certainty.
- "estimated_size_cm": {"length": float, "width": float} - estimated physical real-world dimensions in centimeters.
- "estimated_distance_m": Float - estimated distance from ROV camera in meters (e.g., 0.8, 2.5, 5.0).
- "description": 1 concise sentence describing the object's state, behavior, or condition.
- "hazard_or_conservation_status": Marine status (e.g., "Endangered", "Vulnerable", "Severe Plastic Pollution", "Corrosion Hazard", "Nominal").

Also provide overall "scene_analysis":
- "water_clarity": Assessment of turbidity, optical backscatter, or light penetration.
- "habitat_type": Type of underwater environment (e.g., "Coral Reef", "Pelagic Zone", "Sandy Seabed", "Shipwreck Site").
- "biodiversity_score": Integer 1-10 (1 = dead/sterile, 10 = rich ecosystem).
- "pollution_level": One of ["None", "Low", "Moderate", "High", "Critical"].

Output strictly valid JSON with this format:
{
  "objects": [...],
  "scene_analysis": {...}
}
"""

def clean_json_response(raw_text: str) -> str:
    """Extracts valid JSON substring from LLM response."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # If starting with { and ending with }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

class GeminiUnderwaterDetector:
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_GEMINI_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = "gemini-flash-latest"
        self._init_client()

    def _init_client(self):
        self.client = None
        if not self.api_key:
            return
            
        try:
            # Try new google.genai SDK
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.sdk_type = "genai"
        except Exception:
            try:
                # Fallback to google.generativeai
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self.client = genai_legacy.GenerativeModel(self.model_name)
                self.sdk_type = "generativeai"
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")
                self.client = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self._init_client()

    def detect(self, pil_image: Image.Image, custom_prompt: str = "") -> Dict[str, Any]:
        """
        Runs Gemini Flash underwater detection on a PIL Image.
        """
        if not self.api_key or not self.client:
            return self._mock_detection(pil_image)

        prompt = DETECTION_SYSTEM_PROMPT
        if custom_prompt:
            prompt += f"\nAdditional Operator Instructions: {custom_prompt}"

        try:
            if self.sdk_type == "genai":
                models_to_try = [self.model_name, "gemini-flash-latest", "gemini-3.6-flash", "gemini-2.5-flash-lite"]
                response = None
                last_err = None
                for m in models_to_try:
                    try:
                        response = self.client.models.generate_content(
                            model=m,
                            contents=[pil_image, prompt],
                        )
                        break
                    except Exception as me:
                        last_err = me
                if response is None:
                    raise last_err
                raw_text = response.text
            else:
                response = self.client.generate_content([pil_image, prompt])
                raw_text = response.text

            json_str = clean_json_response(raw_text)
            data = json.loads(json_str)
            return data
        except Exception as e:
            print(f"Gemini API detection error: {e}")
            # Fallback to mock on network/quota error with error notice
            mock_res = self._mock_detection(pil_image)
            mock_res["error"] = str(e)
            return mock_res

    def _mock_detection(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Generates context-aware simulated detections for offline testing and hackathon demos.
        """
        w, h = pil_image.size
        # Sample detection scenarios
        scenarios = [
            # Scenario 1: Coral Reef Biodiversity
            {
                "objects": [
                    {
                        "box_2d": [180, 240, 480, 560],
                        "label": "Green Sea Turtle",
                        "scientific_name": "Chelonia mydas",
                        "category": "marine_life",
                        "confidence": 0.97,
                        "estimated_size_cm": {"length": 92.0, "width": 64.0},
                        "estimated_distance_m": 2.2,
                        "description": "Adult individual swimming above fringing coral shelf.",
                        "hazard_or_conservation_status": "Endangered (IUCN)"
                    },
                    {
                        "box_2d": [560, 120, 860, 420],
                        "label": "Staghorn Coral Colony",
                        "scientific_name": "Acropora cervicornis",
                        "category": "marine_life",
                        "confidence": 0.93,
                        "estimated_size_cm": {"length": 65.0, "width": 48.0},
                        "estimated_distance_m": 1.7,
                        "description": "Intact coral structure with active polyp growth.",
                        "hazard_or_conservation_status": "Critically Endangered"
                    },
                    {
                        "box_2d": [710, 680, 920, 940],
                        "label": "Discarded Plastic Bottle",
                        "scientific_name": "PET Polymer Debris",
                        "category": "debris",
                        "confidence": 0.94,
                        "estimated_size_cm": {"length": 23.0, "width": 7.5},
                        "estimated_distance_m": 1.9,
                        "description": "Sediment-covered beverage bottle posing microplastic risk.",
                        "hazard_or_conservation_status": "Severe Plastic Pollution Hazard"
                    }
                ],
                "scene_analysis": {
                    "water_clarity": "High clarity, minimal backscatter",
                    "habitat_type": "Tropical Coral Reef Edge",
                    "biodiversity_score": 9,
                    "pollution_level": "Low",
                    "is_demo_mode": True
                }
            }
        ]
        return scenarios[0]

def draw_detections(
    pil_image: Image.Image, 
    detection_data: Dict[str, Any], 
    show_boxes: bool = True,
    show_labels: bool = True,
    show_measurements: bool = True,
    fov_deg: float = 75.0
) -> Image.Image:
    """
    Draws modern, high-visibility HUD overlays and bounding boxes on the underwater image.
    """
    img_draw = pil_image.copy()
    draw = ImageDraw.Draw(img_draw, "RGBA")
    w, h = img_draw.size
    
    objects = detection_data.get("objects", [])
    if not objects:
        return img_draw

    for obj in objects:
        box = obj.get("box_2d", [0, 0, 0, 0])
        ymin, xmin, ymax, xmax = box
        
        # Convert 0-1000 scale to pixel coordinates
        if ymax > 1.0 or xmax > 1.0:
            top = int((ymin / 1000.0) * h)
            left = int((xmin / 1000.0) * w)
            bottom = int((ymax / 1000.0) * h)
            right = int((xmax / 1000.0) * w)
        else:
            top = int(ymin * h)
            left = int(xmin * w)
            bottom = int(ymax * h)
            right = int(xmax * w)

        category = obj.get("category", "other")
        cfg = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["other"])
        color_rgb = cfg["color_rgb"]
        icon = cfg["icon"]
        
        # Calculate/refine measurements
        dist_m = obj.get("estimated_distance_m", 2.0)
        size_info = obj.get("estimated_size_cm", {})
        length_cm = size_info.get("length", 0.0)
        width_cm = size_info.get("width", 0.0)
        
        if show_boxes:
            # 1. Soft glowing fill
            fill_color = (*color_rgb, 35)
            draw.rectangle([left, top, right, bottom], fill=fill_color)
            
            # 2. Crisp border
            border_color = (*color_rgb, 240)
            draw.rectangle([left, top, right, bottom], outline=border_color, width=3)
            
            # 3. Corner target crosshairs (High-tech ROV look)
            corner_len = min(18, max(5, int((right - left) * 0.15)))
            # Top-left
            draw.line([(left, top), (left + corner_len, top)], fill=border_color, width=5)
            draw.line([(left, top), (left, top + corner_len)], fill=border_color, width=5)
            # Top-right
            draw.line([(right, top), (right - corner_len, top)], fill=border_color, width=5)
            draw.line([(right, top), (right, top + corner_len)], fill=border_color, width=5)
            # Bottom-left
            draw.line([(left, bottom), (left + corner_len, bottom)], fill=border_color, width=5)
            draw.line([(left, bottom), (left, bottom - corner_len)], fill=border_color, width=5)
            # Bottom-right
            draw.line([(right, bottom), (right - corner_len, bottom)], fill=border_color, width=5)
            draw.line([(right, bottom), (right, bottom - corner_len)], fill=border_color, width=5)

        if show_labels:
            label = obj.get("label", "Object")
            conf = int(obj.get("confidence", 0.9) * 100)
            
            # Build Badge Text
            header_text = f"{icon} {label} [{conf}%]"
            
            meas_text = ""
            if show_measurements:
                if length_cm > 0 and width_cm > 0:
                    meas_text = f" | {length_cm:.0f}?{width_cm:.0f}cm @ {dist_m:.1f}m"
                else:
                    meas_text = f" | Dist: {dist_m:.1f}m"
            
            full_label = header_text + meas_text
            
            # Estimate text background box
            text_len = len(full_label) * 8 + 14
            badge_h = 24
            badge_top = max(0, top - badge_h - 2)
            
            # Badge background
            draw.rectangle(
                [left, badge_top, left + text_len, badge_top + badge_h],
                fill=(10, 25, 47, 220),
                outline=color_rgb,
                width=1
            )
            
            # Badge text
            draw.text(
                (left + 6, badge_top + 4),
                full_label,
                fill=(255, 255, 255, 255)
            )

    return img_draw
