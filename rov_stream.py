"""
Universal ROV Video Stream Manager for SubAquaVision Ultra.
Supports USB (UVC), Wi-Fi (RTSP, HTTP MJPEG, UDP), Phone Camera streams, and HTTP snapshots.
Features threaded frame capture, Real-Time YOLOv8 + ByteTrack Live Tracking (30 FPS),
and background local MJPEG server at port 8504 for smooth 30 FPS browser viewing.
"""

import cv2
import threading
import time
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image

try:
    from local_yolo_detector import RealtimeYOLOTracker
except ImportError:
    from .local_yolo_detector import RealtimeYOLOTracker

def get_available_usb_cameras(max_tested: int = 4) -> List[int]:
    """Detects active camera devices on Windows (testing MSMF default and DSHOW)."""
    available = []
    for i in range(max_tested):
        try:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
                cap.release()
        except Exception:
            pass
    if not available:
        available = [0]
    return available

def find_adb_path() -> Optional[str]:
    import os
    import shutil
    candidates = [
        r"C:\Program Files\ASUS\GlideX\adb.exe",
        r"C:\Program Files\DroidCam\Client\data\obs-plugins\droidcam-obs\adb\adb.exe",
        r"C:\Program Files (x86)\DroidCam\Client\data\obs-plugins\droidcam-obs\adb\adb.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
    ]
    which = shutil.which("adb")
    if which:
        return which
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def setup_usb_adb_forward() -> Tuple[bool, str]:
    """
    Forwards phone video port (8080 for IP Webcam) over physical USB cable
    using ADB so that http://127.0.0.1:8080/video streams directly across the USB cable.
    """
    import subprocess
    import os
    adb_path = find_adb_path()
    if not adb_path:
        return False, "ADB tool nahi mila. Kripya Android Platform Tools install karein."

    try:
        dev = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=5)
        lines = [l.strip() for l in dev.stdout.splitlines() if l.strip() and not l.startswith("List of devices")]
        devices = [l for l in lines if "\tdevice" in l or " device" in l]
        unauthorized = [l for l in lines if "unauthorized" in l]
        
        if unauthorized:
            return False, "⚠️ Phone par 'Allow USB Debugging?' ka prompt aaya hai. 'Always allow' tick karke [OK] dabayein!"
        
        if not devices:
            return False, "⚠️ Phone USB se connect nahi mila. Phone me Developer Options me 'USB Debugging' ON karein aur USB cable reconnect karein."
        
        # Forward port 8080 (IP Webcam HTTP video stream)
        subprocess.run([adb_path, "forward", "tcp:8080", "tcp:8080"], capture_output=True, timeout=3)
        device_id = devices[0].split()[0]
        return True, f"Physical USB Cable Tether Active! (Phone ID: {device_id})"
    except Exception as e:
        return False, f"ADB error: {e}"

def fetch_phone_snapshot(url: str, timeout: float = 3.0) -> Optional[Image.Image]:
    """
    Directly fetches a single high-resolution image frame from a phone camera via HTTP.
    Supports IP Webcam (http://<ip>:8080/shot.jpg or /video) and generic JPEG/PNG URLs.
    """
    import re
    raw = url.strip()
    clean_host = re.sub(r'^[a-zA-Z]+://+', '', raw).split('/')[0]
    if ":" not in clean_host:
        clean_host = f"{clean_host}:8080"
    
    candidates = [
        f"http://{clean_host}/shot.jpg",
        f"http://{clean_host}/photo.jpg"
    ]

    for candidate in candidates:
        try:
            req = urllib.request.Request(
                candidate, 
                headers={'User-Agent': 'Mozilla/5.0 (SubAquaVision Marine ROV Client)'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                data = resp.read()
                if status == 200 and len(data) > 1000:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame_bgr is not None and frame_bgr.shape[0] > 50:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(frame_rgb)
        except Exception:
            continue
            
    return None

class MJPEGStreamServer:
    """
    Lightweight background HTTP server that streams the active ROV stream as MJPEG
    at http://localhost:8504/video_feed for smooth 30 FPS browser rendering.
    Uses port 8504 in SubAquaVisionUltra to eliminate any port conflicts.
    """
    server: Optional[HTTPServer] = None
    thread: Optional[threading.Thread] = None
    active_stream: Optional['ROVStream'] = None
    lock = threading.Lock()
    port = 8504

    @classmethod
    def set_stream(cls, stream: Optional['ROVStream']):
        with cls.lock:
            cls.active_stream = stream
            if cls.server is None and stream is not None:
                cls._start_server()

    @classmethod
    def _start_server(cls):
        class MJPEGHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/video_feed':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    while True:
                        st = MJPEGStreamServer.active_stream
                        if not st or not st.is_running:
                            break
                        # Stream the real-time annotated frame (with live HUD, boxes, and IDs)
                        rgb = st.get_latest_frame_rgb(use_annotated=True)
                        if rgb is not None:
                            try:
                                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                            except Exception:
                                break
                        time.sleep(0.033)  # ~30 FPS
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # suppress console request logging

        try:
            cls.server = HTTPServer(('0.0.0.0', cls.port), MJPEGHandler)
            cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
            cls.thread.start()
        except Exception:
            pass  # port may already be active

class ROVStream:
    """
    Threaded, low-latency video stream handler with integrated Real-Time YOLOv8 + ByteTrack (30 FPS).
    """
    def __init__(self, source: str = "0", width: int = 1280, height: int = 720, conf_threshold: float = 0.25, fov_deg: float = 75.0):
        self.source = str(source).strip()
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_annotated_frame: Optional[np.ndarray] = None
        self.latest_yolo_detections: List[Dict[str, Any]] = []

        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_error = ""

        # Local Real-Time AI Tracking Engine (YOLOv8 + ByteTrack)
        try:
            self.yolo_tracker = RealtimeYOLOTracker(conf_threshold=conf_threshold, fov_deg=fov_deg)
        except Exception as e:
            print(f"YOLO tracker init error: {e}")
            self.yolo_tracker = None

        self.enable_realtime_yolo = True
        self.show_boxes = True

    def set_realtime_yolo(self, enabled: bool):
        self.enable_realtime_yolo = enabled

    def set_show_boxes(self, show: bool):
        self.show_boxes = show

    def set_confidence(self, conf: float):
        if self.yolo_tracker:
            self.yolo_tracker.set_confidence(conf)

    def set_fov(self, fov: float):
        if self.yolo_tracker:
            self.yolo_tracker.set_fov(fov)

    def connect(self) -> bool:
        """Attempts connection to USB port or Wi-Fi RTSP/HTTP URL."""
        self.stop()
        
        # Determine source type
        if self.source.isdigit():
            src = int(self.source)
            self.cap = cv2.VideoCapture(src)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            # Wi-Fi RTSP / HTTP URL
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            self.is_connected = False
            self.last_error = f"Cannot open ROV video source: {self.source}"
            return False

        # Set low buffer size to ensure zero latency (drop old frames)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.source.isdigit():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # CRUCIAL: Verify that the device actually transmits real video frames!
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            self.cap = None
            self.is_connected = False
            self.last_error = f"Camera source '{self.source}' is not transmitting video frames."
            return False

        self.current_frame = test_frame
        self.current_annotated_frame = test_frame
        self.is_connected = True
        self.is_running = True
        self.last_error = ""
        
        # Start background reader thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Register with MJPEG streaming server (port 8504)
        MJPEGStreamServer.set_stream(self)
        
        time.sleep(0.15)
        return True

    def _capture_loop(self):
        """Continuously reads frames and executes real-time YOLO + ByteTrack tracking."""
        last_time = time.time()
        fps_counter = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Real-time YOLOv8 + ByteTrack Processing
            annotated = frame
            dets = []
            if self.enable_realtime_yolo and self.yolo_tracker is not None:
                try:
                    annotated, dets = self.yolo_tracker.process_frame(frame, draw_hud=self.show_boxes)
                except Exception:
                    annotated = frame
                    dets = []

            with self.lock:
                self.current_frame = frame
                self.current_annotated_frame = annotated
                self.latest_yolo_detections = dets
                self.frame_count += 1
                fps_counter += 1

            # Update FPS every 1 second
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = fps_counter / (now - last_time)
                fps_counter = 0
                last_time = now

            time.sleep(0.002)

    def get_latest_frame_rgb(self, use_annotated: bool = True) -> Optional[np.ndarray]:
        """Returns the latest frame in RGB format."""
        with self.lock:
            f = None
            if use_annotated and self.current_annotated_frame is not None:
                f = self.current_annotated_frame
            elif self.current_frame is not None:
                f = self.current_frame
            if f is None:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.current_frame = frame
                        f = frame
            if f is None:
                return None
            frame = f.copy()
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_latest_frame_pil(self, use_annotated: bool = True) -> Optional[Image.Image]:
        """Returns latest frame as PIL Image."""
        rgb = self.get_latest_frame_rgb(use_annotated=use_annotated)
        if rgb is None:
            return None
        return Image.fromarray(rgb)

    def get_latest_detections(self) -> List[Dict[str, Any]]:
        """Returns current list of real-time tracked objects from YOLO + ByteTrack."""
        with self.lock:
            return list(self.latest_yolo_detections)

    def is_standby(self) -> bool:
        """Returns True if the camera stream is inactive or pure black."""
        with self.lock:
            if self.current_frame is None:
                return True
            mean_bgr = np.mean(self.current_frame, axis=(0, 1))
        # Pure black / inactive sensor
        if mean_bgr[0] < 5.0 and mean_bgr[1] < 5.0 and mean_bgr[2] < 5.0:
            return True
        return False

    def stop(self):
        """Stops stream and releases resources."""
        self.is_running = False
        MJPEGStreamServer.set_stream(None)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        with self.lock:
            self.current_frame = None
            self.current_annotated_frame = None
            self.latest_yolo_detections = []

    def __del__(self):
        self.stop()
