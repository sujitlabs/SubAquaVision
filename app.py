"""
SubAquaVision - Underwater Object Detection, ROV Streaming & Measurement System.
Powered by Gemini 2.5 Flash, OpenCV & Streamlit.
Enhanced with Multi-Object Tracking (MOT), Subsea Telemetry HUD, ROV Phone Camera Tether/Wi-Fi Streaming, and Automated Audit Reports.
"""

import os
import sys
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import time
import io
import socket
import plotly.express as px
import plotly.graph_objects as go

from config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION, 
    DEFAULT_GEMINI_MODEL, CATEGORY_CONFIG, 
    DEFAULT_CAMERA_PARAMS, SAMPLE_STREAMS
)
from enhancer import enhance_underwater
from measurer import estimate_object_dimensions, format_measurement_string, estimate_optical_distance
from gemini_detector import GeminiUnderwaterDetector, draw_detections
from rov_stream import ROVStream, get_available_usb_cameras, fetch_phone_snapshot, setup_usb_adb_forward
from tracker import UnderwaterTracker
from telemetry import ROVTelemetry, draw_telemetry_hud
from report_generator import generate_html_mission_report

# Page configuration
st.set_page_config(
    page_title=f"{APP_TITLE} - Underwater AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Futuristic Deep-Sea Mission Control HUD CSS
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
    /* Global Background & Typography */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0c1e38 0%, #050d1a 60%, #020710 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
        letter-spacing: 0.5px;
    }
    
    /* Top Mission Header */
    .mission-header {
        background: linear-gradient(135deg, rgba(15, 30, 55, 0.85) 0%, rgba(8, 18, 35, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 229, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    
    .mission-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.65rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00E5FF 0%, #38BDF8 50%, #00F5A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 0 25px rgba(0, 229, 255, 0.4);
    }
    
    .hud-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .chip-active {
        background: rgba(0, 245, 160, 0.12);
        border: 1px solid #00F5A0;
        color: #00F5A0;
        box-shadow: 0 0 10px rgba(0, 245, 160, 0.25);
    }
    
    .chip-idle {
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid #475569;
        color: #94a3b8;
    }
    
    .chip-cyan {
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid #00E5FF;
        color: #00E5FF;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.25);
    }

    /* Video Monitor Tactical Bezel */
    .tactical-frame {
        position: relative;
        background: #000;
        border: 2px solid #00E5FF;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.25);
        overflow: hidden;
    }
    
    .tactical-bezel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 12px;
        background: rgba(10, 25, 47, 0.9);
        border-bottom: 1px solid rgba(0, 229, 255, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #00E5FF;
    }

    /* Pulse Animations */
    @keyframes pulse-rec {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .rec-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #FF1744;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #FF1744;
        animation: pulse-rec 1.4s infinite ease-in-out;
    }
    
    /* Modern Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(0, 229, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0 16px !important;
        transition: all 0.2s ease-in-out;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #00E5FF !important;
        background: rgba(0, 229, 255, 0.08);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.25) 0%, rgba(3, 105, 161, 0.3) 100%) !important;
        color: #ffffff !important;
        border: 1px solid #00E5FF !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
    }
    
    /* Custom High-Tech Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%) !important;
        border: 1px solid #38bdf8 !important;
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.6) !important;
        border-color: #00E5FF !important;
    }

    .stButton > button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #00E5FF !important;
        color: #00E5FF !important;
        background: rgba(0, 229, 255, 0.1) !important;
    }

    /* Metric HUD Cards */
    .hud-metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(8, 15, 28, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .hud-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .hud-metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Target Cards */
    .target-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(10, 18, 32, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-left: 5px solid #00E5FF;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .target-card:hover {
        transform: translateX(4px);
        border-color: #38bdf8;
    }

    /* Alert Boxes */
    .hazard-alert {
        background: rgba(239, 68, 68, 0.12);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 14px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper to find Local IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "10.200.49.204"

LOCAL_PC_IP = get_local_ip()

# Persistent API Key File
KEY_FILE = os.path.join(os.path.dirname(__file__), "api_key.txt")
saved_key = ""
if os.path.exists(KEY_FILE):
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as kf:
            saved_key = kf.read().strip()
    except Exception:
        pass
if not saved_key:
    saved_key = os.environ.get("GEMINI_API_KEY", "")

# Initialize Session State
if "api_key" not in st.session_state:
    st.session_state.api_key = saved_key
if "detector" not in st.session_state:
    st.session_state.detector = GeminiUnderwaterDetector(api_key=st.session_state.api_key)
if "rov_stream" not in st.session_state:
    st.session_state.rov_stream = None
if "is_connected" not in st.session_state:
    st.session_state.is_connected = False
if "connected_source" not in st.session_state:
    st.session_state.connected_source = ""
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None
if "mission_history" not in st.session_state:
    st.session_state.mission_history = []
if "last_detection_result" not in st.session_state:
    st.session_state.last_detection_result = None
if "last_analyzed_image" not in st.session_state:
    st.session_state.last_analyzed_image = None
if "tracker" not in st.session_state:
    st.session_state.tracker = UnderwaterTracker()
if "telemetry" not in st.session_state:
    st.session_state.telemetry = ROVTelemetry()
if "local_yolo" not in st.session_state:
    try:
        from local_yolo_detector import get_tracker
        st.session_state.local_yolo = get_tracker(conf_threshold=0.25)
    except Exception:
        st.session_state.local_yolo = None

# Function to connect ROV
def connect_camera(source: str):
    if st.session_state.rov_stream is not None:
        st.session_state.rov_stream.stop()
    s = ROVStream(source=source)
    if s.connect():
        st.session_state.rov_stream = s
        st.session_state.is_connected = True
        st.session_state.connected_source = str(source)
        st.session_state["connection_error"] = ""
        return True, "Connected successfully"
    else:
        st.session_state.is_connected = False
        st.session_state.connected_source = ""
        st.session_state["connection_error"] = s.last_error
        return False, s.last_error

# Function to disconnect ROV
def disconnect_camera():
    if st.session_state.rov_stream is not None:
        st.session_state.rov_stream.stop()
        st.session_state.rov_stream = None
    st.session_state.is_connected = False
    st.session_state.connected_source = ""

# Sidebar Controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=600&auto=format&fit=crop&q=60", width='stretch')
    st.title(f"🌊 {APP_TITLE}")
    st.caption(f"{APP_SUBTITLE} (v{APP_VERSION})")
    st.divider()

    # AI Engine Configuration
    st.subheader("⚡ AI Vision Engine")
    ai_engine_mode = st.radio(
        "AI Detection Pipeline",
        [
            "⚡ Local Real-Time AI (YOLOv8 + ByteTrack - 30 FPS)",
            "🤖 Cloud Multimodal AI (Google Gemini Flash)"
        ],
        index=0,
        help="Local AI runs on-device at 30 FPS with continuous ByteTrack. Cloud AI runs multimodal taxonomy on captured frames."
    )
    local_mode_selected = "Local Real-Time" in ai_engine_mode
    is_realtime_local = local_mode_selected and st.session_state.local_yolo is not None

    if local_mode_selected and is_realtime_local:
        st.success("🟢 **Real-Time 30 FPS Mode Active!** Live video continuously tracks objects with ByteTrack IDs.")
        yolo_conf = st.slider("Local AI Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        if st.session_state.rov_stream is not None:
            st.session_state.rov_stream.set_realtime_yolo(True)
            st.session_state.rov_stream.set_confidence(yolo_conf)
    elif local_mode_selected:
        st.warning("⚠️ Local YOLOv8 is unavailable. Install dependencies with `pip install -r requirements.txt`, or select Cloud AI mode.")
        if st.session_state.rov_stream is not None:
            st.session_state.rov_stream.set_realtime_yolo(False)
    else:
        st.info("☁️ **Cloud Gemini Mode:** Uses Gemini 3.5 Flash for deep biological taxonomy on captured frames.")
        if st.session_state.rov_stream is not None:
            st.session_state.rov_stream.set_realtime_yolo(False)

    # Gemini API Configuration
    st.subheader("🤖 Gemini 3.5 Flash API (Optional for Deep Audits)")
    input_api_key = st.text_input(
        "Enter Google Gemini API Key", 
        value=st.session_state.api_key, 
        type="password",
        help="Multimodal zero-shot vision detector powered by Gemini Flash."
    )
    if input_api_key != st.session_state.api_key:
        st.session_state.api_key = input_api_key
        st.session_state.detector.set_api_key(input_api_key)
        if input_api_key:
            try:
                with open(KEY_FILE, "w", encoding="utf-8") as kf:
                    kf.write(input_api_key.strip())
                st.success("API Key saved permanently!")
            except Exception:
                pass
        
    if not st.session_state.api_key:
        st.info("ℹ️ Running in **Interactive Demo Mode**. Add your Gemini API key for live zero-shot inference.")

    st.divider()

    # ROV Connection Settings
    st.subheader("📡 ROV Camera Input")
    
    # PERMANENT CONNECTION STATUS BADGE (NEVER DISAPPEARS)
    if st.session_state.is_connected and st.session_state.rov_stream and st.session_state.rov_stream.is_connected:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 8px; padding: 10px; margin-bottom: 12px; text-align: center;">
            <span style="color: #10b981; font-weight: bold; font-size: 1rem;">🟢 ROV STREAM ONLINE</span><br>
            <span style="color: #a7f3d0; font-size: 0.85rem;">Active Source: <b>{st.session_state.connected_source}</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(100, 116, 139, 0.15); border: 1px solid #64748b; border-radius: 8px; padding: 10px; margin-bottom: 12px; text-align: center;">
            <span style="color: #94a3b8; font-weight: bold; font-size: 1rem;">⚪ ROV STREAM OFFLINE</span><br>
            <span style="color: #64748b; font-size: 0.85rem;">Select camera port & connect</span>
        </div>
        """, unsafe_allow_html=True)

    conn_mode = st.radio("Connection Mode", ["🔌 USB Cable Tether (Phone)", "📶 Wi-Fi (IP Webcam / Phone)", "💻 Laptop / USB Webcam Port"], index=0)
    
    if conn_mode == "🔌 USB Cable Tether (Phone)":
        st.info("💡 **Tether Mode:** Phone ko USB cable se connect karein (USB Debugging ON) aur IP Webcam me 'Start server' dabayein.")
        rov_source = "http://127.0.0.1:8080/video"
    elif conn_mode == "📶 Wi-Fi (IP Webcam / Phone)":
        wifi_phone_ip = st.text_input("Phone IP Address (from IP Webcam):", value="10.200.48.182")
        import re
        clean_ip = re.sub(r'^[a-zA-Z]+://+', '', wifi_phone_ip.strip()).split('/')[0]
        if ":" not in clean_ip:
            clean_ip = f"{clean_ip}:8080"
        rov_source = f"http://{clean_ip}/video"
        st.caption(f"Stream: `{rov_source}`")
    else:
        cam_options = ["Camera Port 0 (Laptop Webcam / USB)", "Camera Port 1 (External USB Camera)", "Camera Port 2", "Custom Port Index"]
        selected_cam_str = st.selectbox("Select Camera Port", cam_options, index=0)
        if selected_cam_str == "Custom Port Index":
            rov_source = st.text_input("Enter Camera Index (e.g. 0, 1, 2)", value="0")
        elif "Port 0" in selected_cam_str:
            rov_source = "0"
        elif "Port 1" in selected_cam_str:
            rov_source = "1"
        else:
            rov_source = "2"

    # Connect / Disconnect Buttons
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🟢 Connect ROV", type="primary", width='stretch'):
            if conn_mode == "🔌 USB Cable Tether (Phone)":
                ok, adb_msg = setup_usb_adb_forward()
                if not ok:
                    st.error(adb_msg)
                else:
                    success, msg = connect_camera(rov_source)
                    if success:
                        st.rerun()
                    else:
                        st.error(f"Failed to stream: {msg}. Make sure 'Start server' is running in IP Webcam!")
            else:
                success, msg = connect_camera(rov_source)
                if success:
                    st.rerun()
                else:
                    st.error(f"Failed: {msg}")
    with col_c2:
        if st.button("🔴 Disconnect", width='stretch'):
            disconnect_camera()
            st.rerun()

    st.divider()

    # Enhancement & HUD Settings
    st.subheader("⚙️ Optical & HUD Settings")
    enhance_mode = st.selectbox("Underwater Color Filter", ["Balanced (Recommended)", "Deep Restoration", "Fast CLAHE", "Raw (No Filter)"], index=0)
    fov_deg = st.slider("Camera Horizontal FOV (°)", min_value=50.0, max_value=120.0, value=DEFAULT_CAMERA_PARAMS["horizontal_fov_deg"], step=1.0)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        show_boxes = st.checkbox("Bounding Boxes", value=True)
        show_labels = st.checkbox("Class Labels", value=True)
    with col_t2:
        show_meas = st.checkbox("Measurements", value=True)
        show_telemetry = st.checkbox("Subsea Telemetry HUD", value=True)

    if st.session_state.local_yolo is not None:
        st.session_state.local_yolo.set_fov(fov_deg)
    if st.session_state.rov_stream is not None:
        st.session_state.rov_stream.set_fov(fov_deg)
        st.session_state.rov_stream.set_show_boxes(show_boxes)
    
    enable_tracking = st.checkbox("🔄 Multi-Object Tracking (MOT)", value=True, help="Assigns persistent IDs to prevent double-counting across video frames.")

# Main Application Layout - Mission Control HUD
st.markdown("""
<div class="mission-header">
    <div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2.2rem; filter: drop-shadow(0 0 10px #00E5FF);">🌊</span>
            <div>
                <h1 class="mission-title">SUBAQUAVISION ULTRA</h1>
                <div style="font-size: 0.88rem; color: #94a3b8; font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 1.2px;">
                    AUTONOMOUS ROV OPTICAL TELEMETRY & MULTIMODAL METROLOGY SYSTEM
                </div>
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span class="hud-chip chip-cyan">⚡ AI: REAL-TIME YOLOv8 + BYTETRACK</span>
        <span class="hud-chip chip-active">🟢 30 FPS LIVE MOT</span>
        <span class="hud-chip chip-cyan">📡 8504 STREAM</span>
    </div>
</div>
""", unsafe_allow_html=True)

def render_grouped_target_cards(objects_list):
    """
    Groups detected targets by object name/species so that schools of fish, 
    multiple plastic items, or repetitive objects show as one consolidated card
    with total count, average dimensions, and distance range.
    """
    if not objects_list:
        st.info("No targets identified.")
        return

    from collections import OrderedDict
    grouped = OrderedDict()
    
    for obj in objects_list:
        raw_label = obj.get("label", "Detected Target").strip()
        key = raw_label.lower()
        if key not in grouped:
            grouped[key] = {
                "display_label": raw_label,
                "category": obj.get("category", "other"),
                "scientific_name": obj.get("scientific_name", ""),
                "description": obj.get("description", ""),
                "hazard_or_conservation_status": obj.get("hazard_or_conservation_status", "Nominal"),
                "items": []
            }
        grouped[key]["items"].append(obj)

    icon_map = {
        "marine_life": "🐠",
        "debris": "🗑️",
        "infrastructure": "⚙️",
        "diver": "🤿",
        "other": "📱"
    }

    for key, data in grouped.items():
        lbl = data["display_label"]
        cat = data["category"]
        sci = data["scientific_name"]
        status = data["hazard_or_conservation_status"]
        icon = icon_map.get(cat, "🔍")
        items = data["items"]
        count = len(items)

        confidences = [int(it.get("confidence", 0.9) * 100) for it in items]
        avg_conf = sum(confidences) // max(1, len(confidences))

        distances = [it.get("estimated_distance_m", 0.0) for it in items if it.get("estimated_distance_m") is not None]
        avg_dist = sum(distances) / max(1, len(distances)) if distances else 0.0
        min_dist = min(distances) if distances else 0.0
        max_dist = max(distances) if distances else 0.0

        lengths = [it.get("estimated_size_cm", {}).get("length", 0.0) for it in items]
        widths = [it.get("estimated_size_cm", {}).get("width", 0.0) for it in items]
        avg_l = sum(lengths) / max(1, len(lengths)) if lengths else 0.0
        avg_w = sum(widths) / max(1, len(widths)) if widths else 0.0
        min_l, max_l = (min(lengths), max(lengths)) if lengths else (0.0, 0.0)

        desc = data["description"]

        if count > 1:
            plural_label = f"{lbl}s" if not lbl.endswith("s") and not lbl.endswith("sh") else lbl
            range_dim_str = f" <small style='color: #94a3b8;'>({min_l:.1f}–{max_l:.1f} cm)</small>" if max_l != min_l else ""
            range_dist_str = f" <small style='color: #94a3b8;'>({min_dist:.2f}–{max_dist:.2f} m)</small>" if max_dist != min_dist else ""

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #00E5FF; border-left: 6px solid #00E5FF; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.4);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: #38bdf8;">{icon} {lbl}</span>
                        <span style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff; padding: 3px 10px; border-radius: 6px; font-size: 0.88rem; font-weight: 800; border: 1px solid #38bdf8;">
                            🔢 Total: {count} {plural_label}
                        </span>
                    </div>
                    <span style="background: rgba(0, 229, 255, 0.2); color: #00E5FF; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold;">
                        {avg_conf}% Avg Match
                    </span>
                </div>
                {f'<div style="margin-top: 4px; font-size: 0.85rem; color: #94a3b8; font-style: italic;">{sci}</div>' if sci else ''}
                <div style="margin-top: 10px; font-size: 0.92rem; color: #e2e8f0; display: flex; flex-wrap: wrap; gap: 14px;">
                    <span>🔢 <b>Count:</b> <code style="color: #38bdf8; font-size: 1rem; font-weight: bold;">{count} individuals</code></span>
                    <span>📏 <b>Avg Size:</b> <code style="color: #4ade80;">{avg_l:.1f} cm × {avg_w:.1f} cm</code>{range_dim_str}</span>
                    <span>📍 <b>Avg Distance:</b> <code style="color: #60a5fa;">{avg_dist:.2f} m</code>{range_dist_str}</span>
                    <span>🏷️ <b>Category:</b> {cat.replace('_', ' ').title()}</span>
                </div>
                {f'<div style="margin-top: 8px; font-size: 0.85rem; color: #cbd5e1;">📝 <i>{desc}</i></div>' if desc else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid #00E5FF; border-left: 5px solid #00E5FF; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: bold; color: #38bdf8;">{icon} {lbl}</span>
                    <span style="background: rgba(0, 229, 255, 0.2); color: #00E5FF; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold;">{avg_conf}% Match</span>
                </div>
                {f'<div style="margin-top: 4px; font-size: 0.85rem; color: #94a3b8; font-style: italic;">{sci}</div>' if sci else ''}
                <div style="margin-top: 8px; font-size: 0.92rem; color: #e2e8f0; display: flex; flex-wrap: wrap; gap: 14px;">
                    <span>📏 <b>Dimensions:</b> <code style="color: #4ade80;">{avg_l:.1f} cm × {avg_w:.1f} cm</code></span>
                    <span>📍 <b>Camera Distance:</b> <code style="color: #60a5fa;">{avg_dist:.2f} m</code></span>
                    <span>🏷️ <b>Category:</b> {cat.replace('_', ' ').title()}</span>
                </div>
                {f'<div style="margin-top: 6px; font-size: 0.85rem; color: #94a3b8;">📝 <i>{desc}</i></div>' if desc else ''}
            </div>
            """, unsafe_allow_html=True)

tabs = st.tabs([
    "🎥 Live ROV Stream", 
    "🖼️ Image & Video Inspector", 
    "📐 Measurement & Geometry", 
    "📊 Mission Analytics & Export",
    "🤖 Ask Marine AI"
])

# ----------------- TAB 0: LIVE ROV STREAM -----------------
with tabs[0]:
    stream = st.session_state.rov_stream
    if stream and stream.is_connected and st.session_state.is_connected:
        st.success(f"🟢 **ROV Live Stream Active** | Source: `{st.session_state.connected_source}` | Frame Rate: `~25-30 FPS`")
        
        # Check for Camera Standby / Pitch Black screen
        if stream.is_standby():
            st.warning("""
            ⚠️ **Camera Sensor Standby:** Video port is open, but frames are black.  
            Please verify the camera is transmitting active video frames!
            """)

        col_v1, col_v2 = st.columns([3, 2])
        
        with col_v1:
            active_src = str(st.session_state.connected_source)
            st.markdown(f"""
            <div class="tactical-frame">
                <div class="tactical-bezel-header">
                    <div><span class="rec-indicator"></span><b>LIVE OPTICAL SENSOR</b> | CAM-FEED 01</div>
                    <div>SRC: <code>{active_src}</code> | 30 FPS</div>
                </div>
                <img src="http://localhost:8504/video_feed" style="width:100%; max-height:460px; object-fit:contain; background:#000;" onerror="this.onerror=null; this.src='{active_src}';">
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                capture_btn = st.button("📸 Capture & Analyze Current Frame", type="primary", width='stretch', key="tab0_capture_btn")
            with col_b2:
                if st.button("🔴 Stop Feed", width='stretch', key="tab0_stop_btn"):
                    disconnect_camera()
                    st.rerun()

            # Dedicated Display of Captured Frame
            if st.session_state.captured_frame is not None:
                st.markdown("---")
                st.markdown("#### 📸 Captured Photo (Frozen for Inspection)")
                st.image(st.session_state.captured_frame, caption="Captured Frame from Video Sensor", width='stretch')

        with col_v2:
            st.subheader("📊 Real-Time AI Detection & Metrology")
            if is_realtime_local:
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #10b981; font-weight: bold; font-size: 1rem;">🟢 LIVE 30 FPS YOLOv8 + BYTETRACK</span><br>
                        <span style="color: #a7f3d0; font-size: 0.85rem;">Continuous On-Device Tracking & Metrology</span>
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.25); color: #10b981; padding: 4px 10px; border-radius: 6px; font-weight: 800; font-family: monospace;">30 FPS</span>
                </div>
                """, unsafe_allow_html=True)
                live_tracked = stream.get_latest_detections()
                if live_tracked:
                    st.markdown(f"#### 🎯 Active Targets in Live View ({len(live_tracked)} tracked):")
                    render_grouped_target_cards(live_tracked)
                else:
                    st.info("🎯 **Live AI Scanning Active:** Point the camera at objects, marine life, or equipment to track them at 30 FPS.")

            if capture_btn:
                latest_frame = stream.get_latest_frame_pil(use_annotated=False)
                if latest_frame is None and stream.cap and stream.cap.isOpened():
                    ret, raw_f = stream.cap.read()
                    if ret and raw_f is not None:
                        latest_frame = Image.fromarray(cv2.cvtColor(raw_f, cv2.COLOR_BGR2RGB))

                if latest_frame is not None:
                    # Apply enhancement filter
                    if enhance_mode != "Raw (No Filter)":
                        method_map = {"Balanced (Recommended)": "balanced", "Deep Restoration": "deep_restoration", "Fast CLAHE": "fast"}
                        enhanced_np = enhance_underwater(latest_frame, method=method_map.get(enhance_mode, "balanced"))
                        img_to_detect = Image.fromarray(enhanced_np)
                    else:
                        img_to_detect = latest_frame
                    
                    if show_telemetry:
                        img_to_detect = draw_telemetry_hud(img_to_detect, st.session_state.telemetry.update())
                    
                    st.session_state.captured_frame = img_to_detect

                    if is_realtime_local and st.session_state.local_yolo is not None:
                        with st.spinner("⚡ Local YOLOv8 analyzing scene at ultra-low latency..."):
                            arr_rgb = np.array(img_to_detect)
                            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
                            _, dets = st.session_state.local_yolo.process_frame(arr_bgr, draw_hud=False)
                            result = {
                                "objects": dets,
                                "scene_analysis": {
                                    "is_underwater": True,
                                    "water_clarity": "Nominal Live Stream",
                                    "habitat_type": "Live Subsea Inspection",
                                    "biodiversity_score": min(10, len(dets) * 2),
                                    "pollution_level": "Active Monitoring",
                                    "is_demo_mode": False
                                }
                            }
                    else:
                        with st.spinner("🤖 Gemini Flash analyzing scene & measuring detected objects..."):
                            result = st.session_state.detector.detect(img_to_detect)
                            if enable_tracking:
                                tracked_objs = st.session_state.tracker.update(result.get("objects", []))
                                result["objects"] = tracked_objs
                            
                    st.session_state.last_detection_result = result
                    st.session_state.last_analyzed_image = img_to_detect

                    st.session_state.mission_history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": str(st.session_state.connected_source),
                        "objects_count": len(result.get("objects", [])),
                        "details": result
                    })
                else:
                    st.error("Could not capture frame from stream. Make sure camera is actively transmitting.")

            if st.session_state.last_detection_result and st.session_state.last_analyzed_image:
                res = st.session_state.last_detection_result
                annotated_img = draw_detections(
                    st.session_state.last_analyzed_image, 
                    res,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_measurements=show_meas,
                    fov_deg=fov_deg
                )
                st.image(annotated_img, caption="AI Annotated HUD with Size & Distance", width='stretch')
                
                # Check environment type
                scene = res.get("scene_analysis", {})
                is_underwater = scene.get("is_underwater", False)
                clarity = str(scene.get("water_clarity", "")).lower()
                hab = str(scene.get("habitat_type", "")).lower()
                
                # If water clarity or habitat explicitly says dry/indoor/air/terrestrial, it is outside water
                if any(w in clarity or w in hab for w in ["dry", "air", "indoor", "room", "terrestrial", "desk", "lab"]):
                    is_underwater = False
                elif any(w in hab for w in ["coral", "reef", "pelagic", "shipwreck", "seabed", "ocean", "subsea", "marine"]):
                    is_underwater = True

                # Check for genuine critical hazards (not nominal everyday items)
                hazards = [
                    o for o in res.get("objects", []) 
                    if any(h in o.get("hazard_or_conservation_status", "").lower() for h in ["critical", "severe", "pollution hazard", "corrosion hazard", "exposed high voltage", "toxic leak", "structural failure"])
                ]

                if is_underwater:
                    if hazards:
                        st.markdown(f"""
                        <div class="hazard-alert">
                            <strong style="color: #FF1744;">⚠️ CRITICAL SUBSEA HAZARD DETECTED:</strong><br>
                            {len(hazards)} environmental/structural risk(s) flagged in this frame. Inspect mission log immediately.
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"""
                    **🌊 Water Clarity:** {scene.get('water_clarity', 'Clear Subsea')}  
                    **🏞️ Marine Habitat:** {scene.get('habitat_type', 'Subsea Zone')}  
                    **🐠 Biodiversity Score:** `{scene.get('biodiversity_score', 0)}/10` | **⚠️ Marine Pollution:** `{scene.get('pollution_level', 'None')}`
                    """)
                else:
                    # Terrestrial / Indoor Room Display (Outside Water)
                    if hazards:
                        st.markdown(f"""
                        <div class="hazard-alert" style="border-left: 5px solid #ff9800; background: rgba(255, 152, 0, 0.1);">
                            <strong style="color: #ff9800;">⚠️ SAFETY / DEBRIS RISK FLAGGED:</strong><br>
                            {len(hazards)} operational item(s) flagged for inspection.
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"""
                    **🏠 Environment:** `Terrestrial / Indoor (Outside Water)`  
                    **📍 Scene Setting:** {scene.get('habitat_type', 'Indoor Workstation / Room')}  
                    **🎯 Objects Detected:** `{len(res.get('objects', []))} items` | **🛡️ Safety Status:** `{scene.get('pollution_level', 'Nominal')}`
                    """)

                # 🎯 Prominent Detected Objects List with Grouped Count & Dimensions
                objects_list = res.get("objects", [])
                if objects_list:
                    st.markdown("#### 🎯 Identified Targets & Metrology (Grouped by Species/Type):")
                    render_grouped_target_cards(objects_list)
    else:
        st.info("ℹ️ **No ROV currently connected.** Connect your camera in the sidebar or use the quick buttons below:")
        
        col_qc1, col_qc2, col_qc3 = st.columns(3)
        with col_qc1:
            if st.button("🔌 Connect Phone via USB Cable (Tether)", width='stretch', type="primary", key="quick_usb_tether"):
                ok, adb_msg = setup_usb_adb_forward()
                if not ok:
                    st.error(adb_msg)
                else:
                    success, msg = connect_camera("http://127.0.0.1:8080/video")
                    if success:
                        st.rerun()
                    else:
                        st.error(f"Failed to stream: {msg}. Make sure 'Start server' is active in IP Webcam!")
        with col_qc2:
            if st.button("📶 Connect Phone via Wi-Fi (IP Webcam)", width='stretch', key="quick_wifi_ip"):
                success, msg = connect_camera("http://10.200.48.182:8080/video")
                if success:
                    st.rerun()
                else:
                    st.error(f"Failed: {msg}. Open the left sidebar '📶 Wi-Fi (IP Webcam / Phone)' to enter your exact Wi-Fi IP.")
        with col_qc3:
            if st.button("💻 Laptop Webcam (Port 0)", width='stretch', key="quick_p0"):
                success, msg = connect_camera("0")
                if success:
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("### 🧪 Or Test with Built-in Demo Underwater Sample:")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        local_sample_dir = os.path.join(os.path.dirname(__file__), "assets", "samples")
        sample_urls = [
            ("Coral Reef with Marine Life", os.path.join(local_sample_dir, "sample_coral_reef.jpg")),
            ("Submerged Shipwreck Infrastructure", os.path.join(local_sample_dir, "sample_shipwreck.jpg")),
            ("Marine Debris & Plastic Cleanup", os.path.join(local_sample_dir, "sample_debris.jpg"))
        ]
        
        sample_prompts = [
            "Sample 1: Focus strictly on Coral Reef & Marine Life species. Do not detect shipwreck or plastic debris.",
            "Sample 2: Focus strictly on Submerged Shipwreck structure & maritime infrastructure. Do not detect marine life or plastic debris.",
            "Sample 3: Focus strictly on Marine Debris, Plastic waste and discarded nets. Do not detect coral reef or marine life."
        ]

        for idx, (s_name, s_url) in enumerate(sample_urls):
            with [col_s1, col_s2, col_s3][idx]:
                st.caption(f"**{s_name}**")
                st.image(s_url, width='stretch')
                if st.button(f"Analyze Sample {idx+1}", key=f"demo_btn_{idx}", width='stretch'):
                    if os.path.exists(s_url):
                        pil_sample = Image.open(s_url).convert("RGB")
                    else:
                        import urllib.request
                        req = urllib.request.Request(s_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req) as resp:
                            pil_sample = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    
                    with st.spinner(f"Analyzing {s_name}..."):
                        method_map = {"Balanced (Recommended)": "balanced", "Deep Restoration": "deep_restoration", "Fast CLAHE": "fast"}
                        enhanced = enhance_underwater(pil_sample, method=method_map.get(enhance_mode, "balanced"))
                        enhanced_pil = Image.fromarray(enhanced)
                        
                        if show_telemetry:
                            telem = st.session_state.telemetry.update()
                            enhanced_pil = draw_telemetry_hud(enhanced_pil, telem)

                        result = st.session_state.detector.detect(enhanced_pil, custom_prompt=sample_prompts[idx])
                        
                        if enable_tracking:
                            result["objects"] = st.session_state.tracker.update(result.get("objects", []))

                        st.session_state.last_detection_result = result
                        st.session_state.last_analyzed_image = enhanced_pil
                        st.session_state.sample_result = result
                        st.session_state.sample_image = enhanced_pil
                        st.session_state.active_sample_name = s_name
                        st.session_state.captured_frame = enhanced_pil
                        st.rerun()

        # Dedicated Display of Sample Analysis Result directly in Tab 0
        if st.session_state.get("sample_result") and st.session_state.get("sample_image"):
            st.markdown("---")
            col_sr_h1, col_sr_h2 = st.columns([3, 1])
            with col_sr_h1:
                st.markdown(f"### 🎯 Results for **{st.session_state.get('active_sample_name', 'Sample')}**")
            with col_sr_h2:
                if st.button("✖️ Clear Sample Result", key="clear_sample_btn"):
                    st.session_state.sample_result = None
                    st.session_state.sample_image = None
                    st.session_state.last_detection_result = None
                    st.session_state.last_analyzed_image = None
                    st.rerun()

            s_res = st.session_state.sample_result
            s_img = st.session_state.sample_image

            col_sr1, col_sr2 = st.columns([3, 2])
            with col_sr1:
                annotated_sample = draw_detections(
                    s_img,
                    s_res,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_measurements=show_meas,
                    fov_deg=fov_deg
                )
                st.image(annotated_sample, caption=f"AI Annotated HUD | {st.session_state.get('active_sample_name')}", width='stretch')
                
                # Check environment
                scene = s_res.get("scene_analysis", {})
                st.markdown(f"""
                **🌊 Water Clarity:** {scene.get('water_clarity', 'Clear Subsea')} | **🏞️ Habitat:** {scene.get('habitat_type', 'Subsea Zone')}  
                **🐠 Biodiversity Score:** `{scene.get('biodiversity_score', 0)}/10` | **⚠️ Marine Pollution:** `{scene.get('pollution_level', 'None')}`
                """)

            with col_sr2:
                st.markdown("#### 🎯 Identified Targets & Metrology (Grouped by Species/Type):")
                s_objects = s_res.get("objects", [])
                render_grouped_target_cards(s_objects)

# ----------------- TAB 1: IMAGE & VIDEO INSPECTOR -----------------
with tabs[1]:
    st.header("🖼️ Underwater Media Inspector")
    st.markdown("Upload pre-recorded underwater images or video footage for multi-object detection and metric measurement.")
    
    uploaded_file = st.file_uploader("Upload Underwater Image or Video", type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"], key="inspector_uploader")
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if "image" in file_type:
            raw_img = Image.open(uploaded_file).convert("RGB")
            
            method_map = {"Balanced (Recommended)": "balanced", "Deep Restoration": "deep_restoration", "Fast CLAHE": "fast"}
            if enhance_mode != "Raw (No Filter)":
                enhanced_arr = enhance_underwater(raw_img, method=method_map.get(enhance_mode, "balanced"))
                processed_img = Image.fromarray(enhanced_arr)
            else:
                processed_img = raw_img
            
            if show_telemetry:
                telem = st.session_state.telemetry.update()
                processed_img = draw_telemetry_hud(processed_img, telem)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("Original vs Enhanced")
                st.image(processed_img, caption="Underwater Enhanced Image", width='stretch')
                col_ib1, col_ib2 = st.columns(2)
                with col_ib1:
                    if st.button("⚡ Run Local YOLOv8", type="primary", width='stretch', key="img_detect_yolo_btn", disabled=st.session_state.local_yolo is None):
                        with st.spinner("Executing Local YOLOv8 + ByteTrack..."):
                            arr_rgb = np.array(processed_img)
                            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
                            if st.session_state.local_yolo is not None:
                                _, dets = st.session_state.local_yolo.process_frame(arr_bgr, draw_hud=False)
                            else:
                                dets = []
                            result = {
                                "objects": dets,
                                "scene_analysis": {
                                    "is_underwater": True,
                                    "water_clarity": "Local High-Speed Scan",
                                    "habitat_type": "Inspected Media Scene",
                                    "biodiversity_score": min(10, len(dets) * 2),
                                    "pollution_level": "Monitored Target",
                                    "is_demo_mode": False
                                }
                            }
                            st.session_state.inspector_result = result
                            st.session_state.inspector_image = processed_img
                            st.session_state.last_detection_result = result
                            st.session_state.last_analyzed_image = processed_img
                with col_ib2:
                    if st.button("🤖 Run Gemini Flash", width='stretch', key="img_detect_btn"):
                        with st.spinner("Detecting marine objects with Gemini Flash..."):
                            result = st.session_state.detector.detect(processed_img)
                            if enable_tracking:
                                result["objects"] = st.session_state.tracker.update(result.get("objects", []))
                            st.session_state.inspector_result = result
                            st.session_state.inspector_image = processed_img
                            st.session_state.last_detection_result = result
                            st.session_state.last_analyzed_image = processed_img
            
            with col_m2:
                st.subheader("📊 Detections & Bounding Boxes")
                if st.session_state.get("inspector_result") and st.session_state.get("inspector_image"):
                    res = st.session_state.inspector_result
                    annotated = draw_detections(
                        st.session_state.inspector_image, 
                        res,
                        show_boxes=show_boxes,
                        show_labels=show_labels,
                        show_measurements=show_meas,
                        fov_deg=fov_deg
                    )
                    st.image(annotated, caption="Annotated Underwater HUD", width='stretch')
                    
                    buf = io.BytesIO()
                    annotated.save(buf, format="JPEG")
                    st.download_button(
                        label="📥 Download Annotated Frame",
                        data=buf.getvalue(),
                        file_name="subaquavision_annotated.jpg",
                        mime="image/jpeg",
                        width='stretch'
                    )
        
        elif "video" in file_type:
            st.info("🎥 Video file loaded. Extracting keyframes for ROV inspection...")
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            vcap = cv2.VideoCapture(tfile.name)
            total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vcap.get(cv2.CAP_PROP_FPS) or 30.0
            
            frame_idx = st.slider("Select Video Frame to Inspect", 0, max(0, total_frames - 1), 0)
            vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vcap.read()
            vcap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                if show_telemetry:
                    telem = st.session_state.telemetry.update()
                    frame_pil = draw_telemetry_hud(frame_pil, telem)

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.image(frame_pil, caption=f"Frame #{frame_idx} (Time: {frame_idx/fps:.2f}s)", width='stretch')
                    if st.button("🚀 Analyze This Frame with AI", key="vid_frame_detect", type="primary", width='stretch'):
                        with st.spinner("Analyzing video frame with Gemini Flash..."):
                            result = st.session_state.detector.detect(frame_pil)
                            if enable_tracking:
                                result["objects"] = st.session_state.tracker.update(result.get("objects", []))
                            st.session_state.inspector_result = result
                            st.session_state.inspector_image = frame_pil
                            st.session_state.last_detection_result = result
                            st.session_state.last_analyzed_image = frame_pil
                with col_v2:
                    if st.session_state.get("inspector_result") and st.session_state.get("inspector_image"):
                        annotated = draw_detections(
                            st.session_state.inspector_image, 
                            st.session_state.inspector_result,
                            show_boxes=show_boxes,
                            show_labels=show_labels,
                            show_measurements=show_meas,
                            fov_deg=fov_deg
                        )
                        st.image(annotated, caption="Annotated Video Frame", width='stretch')

    if uploaded_file is not None and st.session_state.get("inspector_result"):
        st.divider()
        st.subheader("🎯 Detected Objects & Metric Measurements (Grouped by Species/Type)")
        objects = st.session_state.inspector_result.get("objects", [])
        render_grouped_target_cards(objects)
    else:
        st.info("💡 **Media Upload:** Upload an underwater photo or video file above to perform multi-target detection, segmentation, and optical metrology.")

# ----------------- TAB 2: MEASUREMENT & GEOMETRY -----------------
with tabs[2]:
    st.header("📐 Optical Size & Distance Measurement Physics")
    st.markdown("""
    SubAquaVision uses **pinhole perspective geometry** coupled with **underwater refraction compensation (Snell's Law $n=1.333$)** and **Gemini spatial depth reasoning** to estimate real-world target dimensions without needing heavy hardware lasers.
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("🔬 Optical Geometry Model")
        st.markdown(r"""
        The distance $D$ from the ROV camera to the underwater target is given by:
        
        $$D = \frac{f_{px} \cdot W_{real}}{W_{pixel}}$$
        
        Where:
        - $f_{px} = \frac{W_{image}}{2 \cdot \tan(\theta_{apparent} / 2)}$ is the effective underwater focal length in pixels.
        - $\theta_{apparent} = 2 \cdot \arcsin\left(\frac{\sin(\theta_{air} / 2)}{n_{water}}\right)$ accounts for seawater optical refraction through flat dome ports ($n_{water} \approx 1.333$).
        - $W_{pixel}$ is the bounding box pixel span.
        """)
        
    with col_p2:
        st.subheader("🎛️ Interactive Dimension Simulator")
        sim_box_px = st.slider("Target Pixel Width ($W_{px}$)", 20, 1000, 240, step=10)
        sim_dist_m = st.slider("Distance from ROV ($D$ in meters)", 0.3, 10.0, 2.0, step=0.1)
        
        dims = estimate_object_dimensions(
            box_normalized=[100, 100, 100 + sim_box_px, 100 + sim_box_px],
            image_width_px=1920,
            image_height_px=1080,
            distance_m=sim_dist_m,
            fov_deg=fov_deg
        )
        
        st.success(f"**Calculated Object Width:** `{dims['width_cm']:.1f} cm` ({dims['width_cm']/100:.2f} m)")
        st.info(f"**Effective Pixel Scale:** `{dims['width_cm']/sim_box_px:.3f} cm/pixel` at {sim_dist_m}m")

# ----------------- TAB 3: MISSION ANALYTICS & EXPORT -----------------
with tabs[3]:
    st.header("📊 Mission Analytics, Tracking & Audit Report")
    
    mot_summary = st.session_state.tracker.get_summary()
    st.subheader("🔄 Multi-Object Tracking (MOT) Cumulative Stats")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Unique Total Targets", mot_summary.get("total_unique_objects", 0))
    c_m2.metric("Unique Marine Species", mot_summary.get("unique_species_count", 0))
    c_m3.metric("Unique Trash / Debris", mot_summary.get("unique_debris_count", 0))
    c_m4.metric("Active Tracks", mot_summary.get("currently_tracked", 0))
    
    st.divider()

    if st.session_state.last_detection_result:
        res = st.session_state.last_detection_result
        objects = res.get("objects", [])
        
        if objects:
            rows = []
            for obj in objects:
                size = obj.get("estimated_size_cm", {})
                rows.append({
                    "Track ID": f"#{obj.get('track_id', 1)}",
                    "Category": obj.get("category", "other"),
                    "Label": obj.get("label", ""),
                    "Scientific Name": obj.get("scientific_name", ""),
                    "Confidence": f"{int(obj.get('confidence', 0.9)*100)}%",
                    "Length (cm)": size.get("length", 0.0),
                    "Width (cm)": size.get("width", 0.0),
                    "Distance (m)": obj.get("estimated_distance_m", 0.0),
                    "Status": obj.get("hazard_or_conservation_status", "Nominal")
                })
            df = pd.DataFrame(rows)
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                cat_counts = df["Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_pie = px.pie(
                    cat_counts, 
                    names="Category", 
                    values="Count", 
                    title="Detection Category Distribution",
                    hole=0.4,
                    color_discrete_sequence=["#00E676", "#FF1744", "#00E5FF", "#FFEA00", "#D500F9"]
                )
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0f2fe")
                st.plotly_chart(fig_pie, width='stretch')
                
            with col_g2:
                fig_scatter = px.scatter(
                    df,
                    x="Distance (m)",
                    y="Length (cm)",
                    color="Category",
                    size="Length (cm)",
                    hover_name="Label",
                    title="Object Size vs. Camera Distance",
                    color_discrete_sequence=["#00E676", "#FF1744", "#00E5FF"]
                )
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0f2fe")
                st.plotly_chart(fig_scatter, width='stretch')

            st.dataframe(df, width='stretch')
            
            st.markdown("### 📑 Official Mission Export Center")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Export CSV Spreadsheet",
                    data=csv_data,
                    file_name=f"subaquavision_report_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            with col_e2:
                json_data = json.dumps(res, indent=2)
                st.download_button(
                    "📦 Export Raw JSON Metadata",
                    data=json_data,
                    file_name=f"subaquavision_audit_{int(time.time())}.json",
                    mime="application/json",
                    width='stretch'
                )
            with col_e3:
                if st.session_state.last_analyzed_image:
                    html_report = generate_html_mission_report(res, st.session_state.last_analyzed_image)
                    st.download_button(
                        "🖨️ Download Printable PDF/HTML Report",
                        data=html_report.encode('utf-8'),
                        file_name=f"subaquavision_official_report_{int(time.time())}.html",
                        mime="text/html",
                        type="primary",
                        width='stretch'
                    )
        else:
            st.info("No objects recorded in this frame.")
    else:
        st.info("Run an analysis on an ROV stream or image to generate mission analytics.")

# ----------------- TAB 4: ASK MARINE AI -----------------
with tabs[4]:
    st.header("🤖 Marine Vision AI Assistant")
    st.markdown("Ask natural language questions about the current underwater scene, subsea infrastructure integrity, or species behavior.")
    
    if st.session_state.last_analyzed_image is not None:
        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            st.image(st.session_state.last_analyzed_image, caption="Current Inspected Frame", width='stretch')
        
        with col_q2:
            user_question = st.text_input(
                "Ask a question about this underwater frame:", 
                value="Are there any signs of pipeline corrosion, ghost fishing nets, or toxic pollution?"
            )
            if st.button("💬 Inquire AI", type="primary"):
                with st.spinner("Gemini Flash analyzing custom marine query..."):
                    res = st.session_state.detector.detect(
                        st.session_state.last_analyzed_image,
                        custom_prompt=f"Please answer this specific operator question thoroughly: {user_question}"
                    )
                    st.markdown("### 🧠 Marine AI Assessment:")
                    scene = res.get("scene_analysis", {})
                    st.write(f"**Habitat & Conditions:** {scene.get('habitat_type', 'N/A')} ({scene.get('water_clarity', 'N/A')})")
                    for obj in res.get("objects", []):
                        st.markdown(f"- **{obj.get('label')}**: {obj.get('description')} *(Status: {obj.get('hazard_or_conservation_status')})*")
    else:
        st.info("Please capture an ROV frame or upload an underwater image first in Tab 0, Tab 1, or Tab 2.")
