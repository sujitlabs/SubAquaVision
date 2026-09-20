"""
Professional Subsea Mission & Audit Report Generator for SubAquaVision.
Produces print-ready HTML/PDF mission reports with detection tables, metrics, and images.
"""

import base64
import io
import time
from typing import Dict, Any, List
from PIL import Image

def generate_html_mission_report(
    detection_result: Dict[str, Any],
    annotated_image: Image.Image,
    operator_name: str = "Subsea Vision Operator",
    rov_id: str = "ROV-SubAquaVision-01",
    mission_notes: str = "Standard underwater survey & object classification run."
) -> str:
    # Convert image to base64 for embedding
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    objects = detection_result.get("objects", [])
    scene = detection_result.get("scene_analysis", {})

    marine_count = sum(1 for o in objects if o.get("category") == "marine_life")
    debris_count = sum(1 for o in objects if o.get("category") == "debris")
    infra_count = sum(1 for o in objects if o.get("category") == "infrastructure")

    table_rows = ""
    for idx, obj in enumerate(objects, 1):
        size = obj.get("estimated_size_cm", {})
        l_cm = size.get("length", 0.0)
        w_cm = size.get("width", 0.0)
        dist_m = obj.get("estimated_distance_m", 0.0)
        conf = int(obj.get("confidence", 0.9) * 100)
        status = obj.get("hazard_or_conservation_status", "Nominal")

        cat_color = "#00E676" if obj.get("category") == "marine_life" else "#FF1744" if obj.get("category") == "debris" else "#00E5FF"

        table_rows += f"""
        <tr>
            <td><strong>#{idx}</strong></td>
            <td><span style="color: {cat_color}; font-weight: bold;">{obj.get('category', 'other').replace('_', ' ').title()}</span></td>
            <td><strong>{obj.get('label', 'Unknown')}</strong><br><small style="color: #666; font-style: italic;">{obj.get('scientific_name', '')}</small></td>
            <td>{l_cm:.1f} ? {w_cm:.1f} cm</td>
            <td>{dist_m:.2f} m</td>
            <td><span class="badge" style="background-color: #0d213a; color: #fff; padding: 4px 8px; border-radius: 4px;">{conf}%</span></td>
            <td><strong>{status}</strong></td>
            <td><small>{obj.get('description', '')}</small></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SubAquaVision - Subsea Mission Audit Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; color: #1e293b; padding: 30px; margin: 0; }}
        .report-card {{ max-width: 1000px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 35px; }}
        .header {{ border-bottom: 3px solid #00f2fe; padding-bottom: 20px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ margin: 0; color: #0f172a; font-size: 26px; }}
        .header .subtitle {{ color: #64748b; font-size: 14px; margin-top: 5px; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .meta-box {{ background: #f1f5f9; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #0284c7; }}
        .meta-box .label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: bold; }}
        .meta-box .val {{ font-size: 16px; font-weight: bold; color: #0f172a; margin-top: 4px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
        .stat-box {{ text-align: center; padding: 15px; border-radius: 8px; background: #fafafa; border: 1px solid #e2e8f0; }}
        .stat-box .num {{ font-size: 24px; font-weight: bold; }}
        .stat-marine {{ color: #059669; }}
        .stat-debris {{ color: #dc2626; }}
        .stat-infra {{ color: #0284c7; }}
        .annotated-img {{ width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th {{ background: #0f172a; color: #fff; text-align: left; padding: 10px; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        .footer {{ margin-top: 35px; border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 12px; color: #94a3b8; text-align: center; }}
        @media print {{
            body {{ padding: 0; background: #fff; }}
            .report-card {{ box-shadow: none; padding: 10px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <h1>?? SubAquaVision: Subsea Mission Report</h1>
                <div class="subtitle">AI-Powered Underwater Object Detection, Measurement & Audit Log</div>
            </div>
            <div style="text-align: right;">
                <button class="no-print" onclick="window.print()" style="background: #0284c7; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold;">??? Print / Save as PDF</button>
            </div>
        </div>

        <div class="meta-grid">
            <div class="meta-box"><div class="label">Date & Time</div><div class="val">{time.strftime('%Y-%m-%d %H:%M:%S')}</div></div>
            <div class="meta-box"><div class="label">ROV Unit ID</div><div class="val">{rov_id}</div></div>
            <div class="meta-box"><div class="label">Operator</div><div class="val">{operator_name}</div></div>
            <div class="meta-box"><div class="label">AI Model</div><div class="val">Gemini 2.5 Flash</div></div>
        </div>

        <div class="stats-grid">
            <div class="stat-box"><div class="num">{len(objects)}</div><div>Total Detections</div></div>
            <div class="stat-box"><div class="num stat-marine">{marine_count}</div><div>Marine Life</div></div>
            <div class="stat-box"><div class="num stat-debris">{debris_count}</div><div>Debris / Trash</div></div>
            <div class="stat-box"><div class="num stat-infra">{infra_count}</div><div>Infrastructure</div></div>
        </div>

        <h3>?? Annotated ROV HUD Capture</h3>
        <img class="annotated-img" src="data:image/jpeg;base64,{img_b64}" alt="Annotated ROV Frame">

        <h3>?? Detected Targets & Metric Measurements</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Category</th>
                    <th>Object & Taxonomy</th>
                    <th>Size (L ? W)</th>
                    <th>Distance</th>
                    <th>Confidence</th>
                    <th>Status</th>
                    <th>AI Observations</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="footer">
            Generated automatically by SubAquaVision Subsea Inspection Suite &bull; Gemini Multimodal Intelligence
        </div>
    </div>
</body>
</html>
"""
    return html
