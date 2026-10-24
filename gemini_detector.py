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
You are an expert dual-domain computer vision and environmental science AI for subsea ROVs and surface/terrestrial monitoring.
Your mission is to perform high-accuracy object detection, physical measurement estimation, and environmental scene analysis on both underwater footage and dry-land/indoor camera feeds.

Analyze the provided camera frame carefully. Detect ALL visible objects belonging to these categories:
1. "marine_life" (fish, aquatic organisms, sea turtles, corals, or terrestrial animals if on land)
2. "debris" (plastic bottles, cups, packaging waste, ghost nets, cans, tires, discarded objects)
3. "infrastructure" (cables, pipes, electronics, laptops, equipment, tools, furniture, metal/concrete structures)
4. "diver" (human beings, ROV operators, faces, swimmers, divers, persons)
5. "other" (smartphones, household objects, office items, room fixtures, geological formations)

For EVERY detected object, output a JSON object with:
- "box_2d": [ymin, xmin, ymax, xmax] normalized on a scale of 0 to 1000 (integers).
- "label": Clear common name (e.g. "Plastic Bottle", "Human Face", "Subsea Flange", "Laptop Cable").
- "scientific_name": Scientific name, material composition, or technical descriptor (e.g. "Polyethylene Terephthalate", "Homo sapiens", "Copper/PVC").
- "category": One of ["marine_life", "debris", "infrastructure", "diver", "other"].
- "confidence": Float between 0.50 and 0.99 indicating detection certainty.
- "estimated_size_cm": {"length": float, "width": float} - estimated physical real-world dimensions in centimeters.
- "estimated_distance_m": Float - estimated distance from camera sensor in meters (e.g., 0.5, 1.2, 2.5).
- "description": 1 concise sentence describing the object's state, position, or condition.
- "hazard_or_conservation_status": Environmental or structural safety status (e.g., "Plastic Waste", "Nominal", "Corrosion Hazard", "Operator Presence").

Also provide overall "scene_analysis":
- "water_clarity": "Dry / Air / Clear (Terrestrial / Indoor)" if outside water, or water visibility assessment if underwater.
- "habitat_type": Type of environment (e.g., "Indoor / Lab / Dry Room", "Surface / Terrestrial", "Coral Reef", "Pelagic Zone", "Sandy Seabed").
- "biodiversity_score": Integer 1-10 (assessment of living organisms present).
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
        self.model_name = "gemini-3.5-flash-lite"
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
            return self._mock_detection(pil_image, custom_prompt=custom_prompt)

        prompt = DETECTION_SYSTEM_PROMPT
        if custom_prompt:
            prompt += f"\nAdditional Operator Instructions: {custom_prompt}"

        try:
            if self.sdk_type == "genai":
                models_to_try = [self.model_name, "gemini-3.5-flash-lite", "gemini-flash-latest"]
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
            mock_res = self._mock_detection(pil_image, custom_prompt=custom_prompt)
            mock_res["error"] = str(e)
            return mock_res

    def _mock_detection(self, pil_image: Image.Image, custom_prompt: str = "") -> Dict[str, Any]:
        """
        Generates context-aware simulated detections for offline testing and hackathon demos.
        Automatically isolates sample targets (Sample 1 vs 2 vs 3 vs indoor) so they never cross-contaminate.
        """
        arr = np.array(pil_image)
        mean_rgb = np.mean(arr, axis=(0, 1)) # [R, G, B]
        prompt_low = (custom_prompt or "").lower()

        # In water, red light is absorbed strongly: R < 0.70 * B and R < 0.70 * G
        is_water = bool(mean_rgb[0] < 0.70 * mean_rgb[2] and mean_rgb[1] > mean_rgb[0])
        if any(w in prompt_low for w in ["water", "subsea", "coral", "shipwreck", "debris", "marine"]):
            is_water = True
        elif any(w in prompt_low for w in ["room", "indoor", "desk", "outside water", "land", "terrestrial"]):
            is_water = False

        if not is_water:
            # Terrestrial / Indoor Room Scenario
            return {
                "objects": [
                    {
                        "box_2d": [200, 300, 500, 650],
                        "label": "Electronic Equipment & Cables",
                        "scientific_name": "Polycarbonate/Copper Electrical Assembly",
                        "category": "infrastructure",
                        "confidence": 0.95,
                        "estimated_size_cm": {"length": 22.0, "width": 10.0},
                        "estimated_distance_m": 0.8,
                        "description": "Electronic power assembly on dry workstation.",
                        "hazard_or_conservation_status": "Electrical Safety Check"
                    }
                ],
                "scene_analysis": {
                    "is_underwater": False,
                    "water_clarity": "Dry / Air / Clear (Terrestrial / Indoor)",
                    "habitat_type": "Indoor Room / Electronics Setup",
                    "biodiversity_score": 0,
                    "pollution_level": "None",
                    "is_demo_mode": True
                }
            }

        # Subsea Scenario: Strictly differentiate between Sample 1, 2, and 3
        is_sample_2_shipwreck = "shipwreck" in prompt_low or "infrastructure" in prompt_low or "sample 2" in prompt_low or "sample_2" in prompt_low or (mean_rgb[0] > 80 and mean_rgb[1] < 85)
        is_sample_3_debris = "debris" in prompt_low or "plastic" in prompt_low or "sample 3" in prompt_low or "sample_3" in prompt_low or (mean_rgb[1] > 115 and mean_rgb[2] > 125)

        if is_sample_2_shipwreck:
            # 🚢 SAMPLE 2: SUBMERGED SHIPWRECK & INFRASTRUCTURE ONLY
            return {
                "objects": [
                    {
                        "box_2d": [220, 180, 680, 820],
                        "label": "Submerged Shipwreck Hull",
                        "scientific_name": "Corroded Steel Maritime Structure",
                        "category": "infrastructure",
                        "confidence": 0.96,
                        "estimated_size_cm": {"length": 340.0, "width": 180.0},
                        "estimated_distance_m": 3.8,
                        "description": "Historical sunken vessel hull encrusted with maritime growth.",
                        "hazard_or_conservation_status": "Structural Collision Hazard"
                    },
                    {
                        "box_2d": [620, 350, 880, 580],
                        "label": "Subsea Anchor & Chain Assembly",
                        "scientific_name": "Forged Iron Marine Rigging",
                        "category": "infrastructure",
                        "confidence": 0.91,
                        "estimated_size_cm": {"length": 140.0, "width": 75.0},
                        "estimated_distance_m": 2.9,
                        "description": "Heavily oxidized mooring anchor secured into seabed sediment.",
                        "hazard_or_conservation_status": "Entanglement Risk"
                    }
                ],
                "scene_analysis": {
                    "is_underwater": True,
                    "water_clarity": "Moderate, oceanic silt suspension",
                    "habitat_type": "Sunken Wreck / Submerged Structure",
                    "biodiversity_score": 5,
                    "pollution_level": "Low (Corrosion present)",
                    "is_demo_mode": True
                }
            }
        elif is_sample_3_debris:
            # 🗑️ SAMPLE 3: MARINE DEBRIS & POLLUTION CLEANUP ONLY
            return {
                "objects": [
                    {
                        "box_2d": [380, 340, 720, 660],
                        "label": "Discarded Plastic Bottle",
                        "scientific_name": "PET Polymer Marine Debris",
                        "category": "debris",
                        "confidence": 0.95,
                        "estimated_size_cm": {"length": 24.0, "width": 7.8},
                        "estimated_distance_m": 1.4,
                        "description": "Degraded commercial plastic bottle causing microplastic dispersion.",
                        "hazard_or_conservation_status": "Severe Plastic Pollution Hazard"
                    },
                    {
                        "box_2d": [660, 120, 910, 480],
                        "label": "Abandoned Ghost Fishing Net",
                        "scientific_name": "Synthetic Monofilament Mesh",
                        "category": "debris",
                        "confidence": 0.92,
                        "estimated_size_cm": {"length": 95.0, "width": 55.0},
                        "estimated_distance_m": 2.1,
                        "description": "Entangled ghost gear presenting active danger to marine species.",
                        "hazard_or_conservation_status": "Severe Entanglement Hazard"
                    }
                ],
                "scene_analysis": {
                    "is_underwater": True,
                    "water_clarity": "Turbid, suspended particulate debris",
                    "habitat_type": "Coastal Benthic Debris Zone",
                    "biodiversity_score": 2,
                    "pollution_level": "High (Macroplastics detected)",
                    "is_demo_mode": True
                }
            }
        else:
            # 🐠 SAMPLE 1: CORAL REEF & MARINE LIFE ONLY
            return {
                "objects": [
                    {
                        "box_2d": [180, 240, 480, 560],
                        "label": "Green Sea Turtle",
                        "scientific_name": "Chelonia mydas",
                        "category": "marine_life",
                        "confidence": 0.97,
                        "estimated_size_cm": {"length": 92.0, "width": 64.0},
                        "estimated_distance_m": 2.2,
                        "description": "Adult specimen grazing above reef canopy.",
                        "hazard_or_conservation_status": "Endangered (IUCN Red List)"
                    },
                    {
                        "box_2d": [560, 120, 860, 420],
                        "label": "Staghorn Coral Colony",
                        "scientific_name": "Acropora cervicornis",
                        "category": "marine_life",
                        "confidence": 0.93,
                        "estimated_size_cm": {"length": 65.0, "width": 48.0},
                        "estimated_distance_m": 1.7,
                        "description": "Healthy branching coral colony with thriving polyps.",
                        "hazard_or_conservation_status": "Protected Coral Habitat"
                    }
                ],
                "scene_analysis": {
                    "is_underwater": True,
                    "water_clarity": "High clarity, optimal benthic illumination",
                    "habitat_type": "Tropical Coral Reef Edge",
                    "biodiversity_score": 9,
                    "pollution_level": "None",
                    "is_demo_mode": True
                }
            }

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
                    meas_text = f" | {length_cm:.0f}×{width_cm:.0f}cm @ {dist_m:.2f}m"
                else:
                    meas_text = f" | Dist: {dist_m:.2f}m"
            
            full_label = header_text + meas_text
            
            # Load dynamic TrueType font based on image resolution
            target_font_size = max(18, min(42, int(h / 32)))
            hud_font = None
            font_candidates = [
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\tahoma.ttf"
            ]
            for fc in font_candidates:
                if os.path.exists(fc):
                    try:
                        hud_font = ImageFont.truetype(fc, target_font_size)
                        break
                    except Exception:
                        pass
            if hud_font is None:
                try:
                    hud_font = ImageFont.load_default(size=target_font_size)
                except Exception:
                    hud_font = ImageFont.load_default()
            
            # Calculate exact text dimensions
            try:
                tb = draw.textbbox((0, 0), full_label, font=hud_font)
                text_w = tb[2] - tb[0]
                text_h = tb[3] - tb[1]
            except Exception:
                text_w = len(full_label) * int(target_font_size * 0.6)
                text_h = target_font_size + 4

            badge_h = text_h + 10
            badge_w = text_w + 16
            
            # Position badge above box, or just inside top if near image edge
            if top - badge_h - 4 >= 0:
                badge_top = top - badge_h - 4
            else:
                badge_top = top + 4
                
            badge_left = max(2, min(left, w - badge_w - 2))
            
            # Badge background with high contrast and rounded look
            draw.rectangle(
                [badge_left, badge_top, badge_left + badge_w, badge_top + badge_h],
                fill=(10, 20, 35, 235),
                outline=color_rgb,
                width=2
            )
            
            # Badge text (Bold, crisp white)
            draw.text(
                (badge_left + 8, badge_top + 4),
                full_label,
                fill=(255, 255, 255, 255),
                font=hud_font
            )

    return img_draw
