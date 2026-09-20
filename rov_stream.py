"""
Universal ROV Video Stream Manager.
Supports USB (UVC), Wi-Fi (RTSP, HTTP MJPEG, UDP), and video file inputs.
Features threaded frame capture for zero-latency real-time viewing.
"""

import cv2
import threading
import time
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

def get_available_usb_cameras(max_tested: int = 5) -> List[int]:
    """
    Scans for connected USB/UVC cameras.
    Returns list of working camera indices (e.g., [0, 1]).
    """
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # DSHOW for fast Windows detection
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

class ROVStream:
    """
    Threaded, low-latency video stream handler for ROV feeds.
    """
    def __init__(self, source: str = "0", width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_error = ""

    def connect(self) -> bool:
        """Attempts connection to USB port or Wi-Fi RTSP/HTTP URL."""
        self.stop()
        
        # Determine source type
        if str(self.source).isdigit():
            # USB Camera index
            src = int(self.source)
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            # Wi-Fi RTSP / HTTP URL
            src = str(self.source).strip()
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            self.is_connected = False
            self.last_error = f"Cannot open ROV video source: {self.source}"
            return False

        # Set low buffer size to ensure zero latency (drop old frames)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if str(self.source).isdigit():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.is_connected = True
        self.is_running = True
        self.last_error = ""
        
        # Start background reader thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        """Continuously reads frames in the background to prevent buffer buildup."""
        last_time = time.time()
        fps_counter = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.current_frame = frame
                self.frame_count += 1
                fps_counter += 1

            # Update FPS every 1 second
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = fps_counter / (now - last_time)
                fps_counter = 0
                last_time = now

            time.sleep(0.001)

    def get_latest_frame_rgb(self) -> Optional[np.ndarray]:
        """Returns the latest captured frame in RGB format."""
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_latest_frame_pil(self) -> Optional[Image.Image]:
        """Returns latest frame as PIL Image."""
        rgb = self.get_latest_frame_rgb()
        if rgb is None:
            return None
        return Image.fromarray(rgb)

    def stop(self):
        """Stops stream and releases resources."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        with self.lock:
            self.current_frame = None

    def __del__(self):
        self.stop()
