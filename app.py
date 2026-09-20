"""
SubAquaVision - Underwater Object Detection, ROV Streaming & Measurement System.
Powered by Gemini 2.5 Flash, OpenCV & Streamlit.
Enhanced with Multi-Object Tracking (MOT), Subsea Telemetry HUD, and Automated Audit Reports.
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
from rov_stream import ROVStream, get_available_usb_cameras
from tracker import UnderwaterTracker
from telemetry import ROVTelemetry, draw_telemetry_hud
from report_generator import generate_html_mission_report

# Page configuration
st.set_page_config(
    page_title=f"{APP_TITLE} - Underwater AI",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ocean Theme CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #071322 0%, #0d213a 100%);
        color: #e0f2fe;
    }
    .stApp {
        background-color: #050e1a;
    }
    .metric-card {
        background: rgba(13, 33, 58, 0.85);
        border: 1px solid #1e3a8a;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .detection-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .badge-marine { background-color: rgba(0, 230, 118, 0.2); border: 1px solid #00E676; color: #00E676; }
    .badge-debris { background-color: rgba(255, 23, 68, 0.2); border: 1px solid #FF1744; color: #FF1744; }
    .badge-infra  { background-color: rgba(0, 229, 255, 0.2); border: 1px solid #00E5FF; color: #00E5FF; }
    .badge-diver  { background-color: rgba(255, 234, 0, 0.2); border: 1px solid #FFEA00; color: #FFEA00; }
    .hazard-alert {
        background: rgba(255, 23, 68, 0.15);
        border: 2px solid #FF1744;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 15px;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

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

# Sidebar Controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=600&auto=format&fit=crop&q=60", width='stretch')
    st.title(f"?? {APP_TITLE}")
    st.caption(f"{APP_SUBTITLE} (v{APP_VERSION})")
    st.divider()

    # Gemini API Configuration
    st.subheader("?? Gemini 2.5 Flash API")
    input_api_key = st.text_input(
        "Enter Google Gemini API Key", 
        value=st.session_state.api_key, 
        type="password",
        help="Zero-shot multimodal vision model is powered by Gemini Flash."
    )
    if input_api_key != st.session_state.api_key:
        st.session_state.api_key = input_api_key
        st.session_state.detector.set_api_key(input_api_key)
        if input_api_key:
            try:
                with open(KEY_FILE, "w", encoding="utf-8") as kf:
                    kf.write(input_api_key.strip())
                st.success("API Key saved permanently to disk!")
            except Exception as e:
                st.success("API Key updated for this session!")
        
    if not st.session_state.api_key:
        st.info("?? Running in **Interactive Demo Mode**. Add your Gemini API key for live zero-shot inference.")

    st.divider()

    # ROV Connection Settings
    st.subheader("??? ROV Camera Input")
    conn_mode = st.radio("Connection Protocol", ["USB / UVC Camera", "Wi-Fi (RTSP / HTTP Stream)"], index=0)
    
    if conn_mode == "USB / UVC Camera":
        usb_cameras = get_available_usb_cameras(max_tested=3)
        if usb_cameras:
            cam_options = [f"Camera Port {i}" for i in usb_cameras]
            selected_cam_str = st.selectbox("Select Detected USB Camera", cam_options, index=0)
            rov_source = str(usb_cameras[cam_options.index(selected_cam_str)])
        else:
            rov_source = st.text_input("USB Camera Index", value="0")
    else:
        preset_stream = st.selectbox("Stream Presets", list(SAMPLE_STREAMS.keys()), index=1)
        rov_source = st.text_input("Wi-Fi Video URL", value=SAMPLE_STREAMS[preset_stream])

    # Connect / Disconnect Buttons
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Connect ROV", width='stretch'):
            if st.session_state.rov_stream is not None:
                st.session_state.rov_stream.stop()
            stream = ROVStream(source=rov_source)
            if stream.connect():
                st.session_state.rov_stream = stream
                st.success(f"Connected to ROV ({rov_source})")
            else:
                st.error(f"Failed to connect: {stream.last_error}")
    with col_c2:
        if st.button("Disconnect", width='stretch'):
            if st.session_state.rov_stream is not None:
                st.session_state.rov_stream.stop()
                st.session_state.rov_stream = None
                st.info("ROV Disconnected")

    st.divider()

    # Enhancement & HUD Settings
    st.subheader("?? Optical & HUD Settings")
    enhance_mode = st.selectbox("Underwater Color Filter", ["Balanced (Recommended)", "Deep Restoration", "Fast CLAHE", "Raw (No Filter)"], index=0)
    fov_deg = st.slider("Camera Horizontal FOV (?)", min_value=50.0, max_value=120.0, value=DEFAULT_CAMERA_PARAMS["horizontal_fov_deg"], step=1.0)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        show_boxes = st.checkbox("Bounding Boxes", value=True)
        show_labels = st.checkbox("Class Labels", value=True)
    with col_t2:
        show_meas = st.checkbox("Measurements", value=True)
        show_telemetry = st.checkbox("Subsea Telemetry HUD", value=True)
    
    enable_tracking = st.checkbox("?? Multi-Object Tracking (MOT)", value=True, help="Assigns persistent IDs to prevent double-counting across video frames.")

# Main Application Layout
st.title("?? SubAquaVision: Underwater Detection & Measurement")
st.markdown("Universal ROV Camera Stream, Zero-Shot Multimodal Detection, Size/Distance Estimation & Audit Reports")

tabs = st.tabs([
    "?? Live ROV Stream", 
    "??? Image & Video Inspector", 
    "?? Measurement & Geometry", 
    "?? Mission Analytics & Export",
    "?? Ask Marine AI"
])

# ----------------- TAB 1: LIVE ROV STREAM -----------------
with tabs[0]:
    st.header("?? Live ROV Optical Feed")
    
    stream = st.session_state.rov_stream
    if stream and stream.is_connected:
        st.success(f"?? **ROV Live Stream Active** | Source: `{stream.source}` | FPS: `{stream.fps:.1f}`")
        
        col_v1, col_v2 = st.columns([3, 2])
        
        with col_v1:
            frame_placeholder = st.empty()
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                capture_btn = st.button("?? Capture & Analyze Current Frame", type="primary", width='stretch')
            with btn_col2:
                stream_preview = st.toggle("Live Preview Feed", value=True)
            
            latest_frame = stream.get_latest_frame_pil()
            if latest_frame is not None:
                if enhance_mode != "Raw (No Filter)":
                    method_map = {"Balanced (Recommended)": "balanced", "Deep Restoration": "deep_restoration", "Fast CLAHE": "fast"}
                    enhanced_np = enhance_underwater(latest_frame, method=method_map.get(enhance_mode, "balanced"))
                    display_frame = Image.fromarray(enhanced_np)
                else:
                    display_frame = latest_frame
                
                # Apply telemetry HUD if enabled
                if show_telemetry:
                    telem_data = st.session_state.telemetry.update()
                    display_frame = draw_telemetry_hud(display_frame, telem_data)

                frame_placeholder.image(display_frame, caption=f"ROV Optical Feed (Live) - {display_frame.width}x{display_frame.height}px", width='stretch')
            else:
                frame_placeholder.warning("Waiting for ROV video stream packets...")

        with col_v2:
            st.subheader("?? Real-Time Analysis Output")
            if capture_btn and latest_frame is not None:
                with st.spinner("Gemini Flash analyzing underwater scene & measuring objects..."):
                    img_to_detect = display_frame
                    result = st.session_state.detector.detect(img_to_detect)
                    
                    # Apply Multi-Object Tracking if enabled
                    if enable_tracking:
                        tracked_objs = st.session_state.tracker.update(result.get("objects", []))
                        result["objects"] = tracked_objs
                        
                    st.session_state.last_detection_result = result
                    st.session_state.last_analyzed_image = img_to_detect
                    
                    st.session_state.mission_history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": str(stream.source),
                        "objects_count": len(result.get("objects", [])),
                        "details": result
                    })

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
                
                # Check for critical hazards
                hazards = [o for o in res.get("objects", []) if "hazard" in o.get("hazard_or_conservation_status", "").lower() or "severe" in o.get("hazard_or_conservation_status", "").lower() or "plastic" in o.get("hazard_or_conservation_status", "").lower()]
                if hazards:
                    st.markdown(f"""
                    <div class="hazard-alert">
                        <strong style="color: #FF1744;">?? CRITICAL SUBSEA HAZARD DETECTED:</strong><br>
                        {len(hazards)} environmental/structural risk(s) flagged in this frame. Inspect mission log immediately.
                        
                    </div>
                    """, unsafe_allow_html=True)
                
                scene = res.get("scene_analysis", {})
                st.markdown(f"""
                **?? Water Clarity:** {scene.get('water_clarity', 'N/A')}  
                **??? Habitat:** {scene.get('habitat_type', 'N/A')}  
                **?? Biodiversity Score:** `{scene.get('biodiversity_score', 0)}/10` | **?? Pollution:** `{scene.get('pollution_level', 'None')}`
                """)
    else:
        st.info("?? **No ROV currently connected.** Connect your USB ROV camera or enter a Wi-Fi RTSP stream in the sidebar.")
        
        st.markdown("### ?? Or Test with Built-in Demo Underwater Sample:")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        local_sample_dir = os.path.join(os.path.dirname(__file__), "assets", "samples")
        sample_urls = [
            ("Coral Reef with Marine Life", os.path.join(local_sample_dir, "sample_coral_reef.jpg")),
            ("Submerged Shipwreck Infrastructure", os.path.join(local_sample_dir, "sample_shipwreck.jpg")),
            ("Marine Debris & Plastic Cleanup", os.path.join(local_sample_dir, "sample_debris.jpg"))
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
                    
                    with st.spinner("Processing underwater imagery..."):
                        method_map = {"Balanced (Recommended)": "balanced", "Deep Restoration": "deep_restoration", "Fast CLAHE": "fast"}
                        enhanced = enhance_underwater(pil_sample, method=method_map.get(enhance_mode, "balanced"))
                        enhanced_pil = Image.fromarray(enhanced)
                        
                        if show_telemetry:
                            telem = st.session_state.telemetry.update()
                            enhanced_pil = draw_telemetry_hud(enhanced_pil, telem)

                        result = st.session_state.detector.detect(enhanced_pil)
                        
                        if enable_tracking:
                            result["objects"] = st.session_state.tracker.update(result.get("objects", []))

                        st.session_state.last_detection_result = result
                        st.session_state.last_analyzed_image = enhanced_pil
                        st.rerun()

# ----------------- TAB 2: IMAGE & VIDEO INSPECTOR -----------------
with tabs[1]:
    st.header("??? Underwater Media Inspector")
    st.markdown("Upload pre-recorded underwater images or video footage for multi-object detection and metric measurement.")
    
    uploaded_file = st.file_uploader("Upload Underwater Image or Video", type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"])
    
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
                if st.button("?? Run Gemini Flash Detection", type="primary", width='stretch'):
                    with st.spinner("Detecting marine objects and calculating measurements..."):
                        result = st.session_state.detector.detect(processed_img)
                        if enable_tracking:
                            result["objects"] = st.session_state.tracker.update(result.get("objects", []))
                        st.session_state.last_detection_result = result
                        st.session_state.last_analyzed_image = processed_img
            
            with col_m2:
                st.subheader("?? Detections & Bounding Boxes")
                if st.session_state.last_detection_result and st.session_state.last_analyzed_image:
                    res = st.session_state.last_detection_result
                    annotated = draw_detections(
                        st.session_state.last_analyzed_image, 
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
                        label="?? Download Annotated Frame",
                        data=buf.getvalue(),
                        file_name="subaquavision_annotated.jpg",
                        mime="image/jpeg",
                        width='stretch'
                    )
        
        elif "video" in file_type:
            st.info("?? Video file loaded. Extracting keyframes for ROV inspection...")
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            vcap = cv2.VideoCapture(tfile.name)
            total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vcap.get(cv2.CAP_PROP_FPS) or 30.0
            
            frame_idx = st.slider("Select Video Frame to Inspect", 0, total_frames - 1, 0)
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
                    if st.button("?? Analyze This Frame with AI", key="vid_frame_detect", type="primary", width='stretch'):
                        with st.spinner("Analyzing video frame with Gemini Flash..."):
                            result = st.session_state.detector.detect(frame_pil)
                            if enable_tracking:
                                result["objects"] = st.session_state.tracker.update(result.get("objects", []))
                            st.session_state.last_detection_result = result
                            st.session_state.last_analyzed_image = frame_pil
                with col_v2:
                    if st.session_state.last_detection_result and st.session_state.last_analyzed_image:
                        annotated = draw_detections(
                            st.session_state.last_analyzed_image, 
                            st.session_state.last_detection_result,
                            show_boxes=show_boxes,
                            show_labels=show_labels,
                            show_measurements=show_meas,
                            fov_deg=fov_deg
                        )
                        st.image(annotated, caption="Annotated Video Frame", width='stretch')

    if st.session_state.last_detection_result:
        st.divider()
        st.subheader("?? Detected Objects & Metric Measurements")
        
        objects = st.session_state.last_detection_result.get("objects", [])
        if not objects:
            st.info("No objects detected in this scene.")
        else:
            cols = st.columns(min(3, len(objects)))
            for idx, obj in enumerate(objects):
                col = cols[idx % len(cols)]
                category = obj.get("category", "other")
                cfg = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["other"])
                badge_class = f"badge-{category[:5]}"
                
                dist_m = obj.get("estimated_distance_m", 2.0)
                size_info = obj.get("estimated_size_cm", {})
                w_cm = size_info.get("width", 0.0)
                l_cm = size_info.get("length", 0.0)
                track_tag = f" [Track #{obj.get('track_id', idx+1)}]" if 'track_id' in obj else ""

                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {cfg['color_hex']};">
                        <span class="detection-badge {badge_class}">{cfg['icon']} {cfg['name']}{track_tag}</span>
                        <h4 style="margin: 8px 0 4px 0; color: #fff;">{obj.get('label', 'Unknown Object')}</h4>
                        <p style="font-style: italic; font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">{obj.get('scientific_name', '')}</p>
                        <p><strong>?? Size:</strong> {l_cm:.1f} cm ? {w_cm:.1f} cm</p>
                        <p><strong>?? Camera Distance:</strong> {dist_m:.2f} meters</p>
                        <p><strong>? Confidence:</strong> {int(obj.get('confidence', 0.9)*100)}%</p>
                        <p><strong>?? Status:</strong> <code>{obj.get('hazard_or_conservation_status', 'Nominal')}</code></p>
                        <p style="font-size: 0.85rem; color: #cbd5e1;">{obj.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ----------------- TAB 3: MEASUREMENT & GEOMETRY -----------------
with tabs[2]:
    st.header("?? Optical Size & Distance Measurement Physics")
    st.markdown("""
    SubAquaVision uses **pinhole perspective geometry** coupled with **underwater refraction compensation (Snell's Law $n=1.333$)** and **Gemini spatial depth reasoning** to estimate real-world target dimensions without needing heavy hardware lasers.
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("?? Optical Geometry Model")
        st.markdown(r"""
        The distance $D$ from the ROV camera to the underwater target is given by:
        
        $$D = rac{f_{px} \cdot W_{real}}{W_{pixel}}$$
        
        Where:
        - $f_{px} = rac{W_{image}}{2 \cdot 	an(	heta_{apparent} / 2)}$ is the effective underwater focal length in pixels.
        - $	heta_{apparent} = 2 \cdot rcsin\left(rac{\sin(	heta_{air} / 2)}{n_{water}}
ight)$ accounts for seawater optical refraction through flat dome ports ($n_{water} pprox 1.333$).
        - $W_{pixel}$ is the bounding box pixel span.
        """)
        
    with col_p2:
        st.subheader("??? Interactive Dimension Simulator")
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

# ----------------- TAB 4: MISSION ANALYTICS & EXPORT -----------------
with tabs[3]:
    st.header("?? Mission Analytics, Tracking & Audit Report")
    
    # Unique MOT summary
    mot_summary = st.session_state.tracker.get_summary()
    st.subheader("?? Multi-Object Tracking (MOT) Cumulative Stats")
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
            
            st.markdown("### ?? Official Mission Export Center")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "?? Export CSV Spreadsheet",
                    data=csv_data,
                    file_name=f"subaquavision_report_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            with col_e2:
                json_data = json.dumps(res, indent=2)
                st.download_button(
                    "?? Export Raw JSON Metadata",
                    data=json_data,
                    file_name=f"subaquavision_audit_{int(time.time())}.json",
                    mime="application/json",
                    width='stretch'
                )
            with col_e3:
                # Generate print-ready HTML/PDF report
                if st.session_state.last_analyzed_image:
                    html_report = generate_html_mission_report(res, st.session_state.last_analyzed_image)
                    st.download_button(
                        "??? Download Printable PDF/HTML Report",
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

# ----------------- TAB 5: ASK MARINE AI -----------------
with tabs[4]:
    st.header("?? Marine Vision AI Assistant")
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
            if st.button("?? Inquire AI", type="primary"):
                with st.spinner("Gemini Flash analyzing custom marine query..."):
                    res = st.session_state.detector.detect(
                        st.session_state.last_analyzed_image,
                        custom_prompt=f"Please answer this specific operator question thoroughly: {user_question}"
                    )
                    st.markdown("### ?? Marine AI Assessment:")
                    scene = res.get("scene_analysis", {})
                    st.write(f"**Habitat & Conditions:** {scene.get('habitat_type', 'N/A')} ({scene.get('water_clarity', 'N/A')})")
                    for obj in res.get("objects", []):
                        st.markdown(f"- **{obj.get('label')}**: {obj.get('description')} *(Status: {obj.get('hazard_or_conservation_status')})*")
    else:
        st.info("Please capture an ROV frame or upload an underwater image first in Tab 1 or Tab 2.")
