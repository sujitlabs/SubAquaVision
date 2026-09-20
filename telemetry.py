"""
Subsea ROV Telemetry & Heads-Up Display (HUD) Simulator.
Provides Depth, Water Temperature, Heading, and ROV status telemetry.
"""

import time
import math
import random
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

class ROVTelemetry:
    def __init__(self, start_depth_m: float = 14.5, water_temp_c: float = 24.2):
        self.depth_m = start_depth_m
        self.water_temp_c = water_temp_c
        self.heading_deg = 184
        self.battery_pct = 94
        self.thruster_load = 42
        self.last_update = time.time()

    def update(self) -> Dict[str, Any]:
        """Simulates realistic micro-fluctuations in ROV subsea telemetry."""
        now = time.time()
        dt = now - self.last_update
        if dt > 1.0:
            self.depth_m = round(max(0.5, self.depth_m + random.uniform(-0.15, 0.15)), 2)
            self.heading_deg = (self.heading_deg + random.choice([-1, 0, 1])) % 360
            self.water_temp_c = round(self.water_temp_c + random.uniform(-0.02, 0.02), 1)
            self.thruster_load = int(max(10, min(95, self.thruster_load + random.randint(-4, 4))))
            self.last_update = now

        return {
            "depth_m": self.depth_m,
            "heading_deg": self.heading_deg,
            "water_temp_c": self.water_temp_c,
            "battery_pct": self.battery_pct,
            "thruster_load_pct": self.thruster_load,
            "status": "ARMED / STABILIZED",
            "pressure_bar": round(1.0 + (self.depth_m / 10.0), 2)
        }

def draw_telemetry_hud(pil_image: Image.Image, telemetry: Dict[str, Any]) -> Image.Image:
    """Draws realistic military/scientific ROV telemetry HUD overlay."""
    img_draw = pil_image.copy()
    draw = ImageDraw.Draw(img_draw, "RGBA")
    w, h = img_draw.size

    cyan = (0, 242, 254, 230)
    bg_dark = (7, 19, 34, 180)

    # 1. Top HUD Bar
    draw.rectangle([0, 0, w, 32], fill=bg_dark)
    top_text = f"ROV TELEMETRY | DEPTH: {telemetry.get('depth_m', 0):.1f}m | HEAD: {telemetry.get('heading_deg', 0)}? | TEMP: {telemetry.get('water_temp_c', 0):.1f}?C | PRESS: {telemetry.get('pressure_bar', 1):.2f} bar"
    draw.text((15, 8), top_text, fill=cyan)

    # 2. Bottom HUD Bar
    draw.rectangle([0, h - 30, w, h], fill=bg_dark)
    bot_text = f"SYS: {telemetry.get('status', 'ONLINE')} | BATT: {telemetry.get('battery_pct', 100)}% | THRUST: {telemetry.get('thruster_load_pct', 0)}%"
    draw.text((15, h - 22), bot_text, fill=(0, 230, 118, 240))

    # 3. Center Target Crosshair
    cx, cy = w // 2, h // 2
    r = 25
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 242, 254, 80), width=1)
    draw.line([(cx - r - 10, cy), (cx + r + 10, cy)], fill=(0, 242, 254, 80), width=1)
    draw.line([(cx, cy - r - 10), (cx, cy + r + 10)], fill=(0, 242, 254, 80), width=1)

    return img_draw
