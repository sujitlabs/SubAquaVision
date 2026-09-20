# ?? SubAquaVision: Underwater Object Detection & Measurement Software

An intelligent, pure-software system designed for **underwater object detection**, **real-time ROV camera streaming (USB & Wi-Fi)**, **size and distance measurement**, and **marine environmental analysis** powered by **Google Gemini 2.5 Flash** and **OpenCV**.

---

## ?? Key Features

1. **High Speed & Low Latency:**
   - Powered by Gemini 2.5 Flash for near-instant multimodal zero-shot vision.
   - Zero-latency threaded video buffer for smooth live ROV streaming.
2. **High Accuracy Detection:**
   - 2D Bounding Boxes with category color-coding:
     - ?? **Marine Life:** Species identification, taxonomy & health.
     - ?? **Marine Debris / Trash:** Plastics, ghost nets, corrosion hazards.
     - ?? **Subsea Infrastructure:** Pipelines, cables, shipwrecks.
     - ?? **Divers & Human Activity:** Divers, dive gear, safety markers.
3. **Measurement Engine (Size & Distance):**
   - Real-world physical dimensions (Length $\times$ Width in cm/meters).
   - Distance from ROV camera in meters using optical perspective geometry and refraction compensation ($n_{water} = 1.333$).
4. **Universal ROV Connectivity:**
   - **USB / UVC Cameras:** Plug-and-play direct webcam selector.
   - **Wi-Fi / RTSP / HTTP:** Connect to BlueROV2, ESP32-CAM, Raspberry Pi, or subsea IP cameras.
5. **Underwater Optical Enhancement:**
   - Red-channel compensation, White Balance, and CLAHE (Contrast Limited Adaptive Histogram Equalization) to remove water turbidity and color cast.
6. **Mission Reports & Analytics:**
   - Interactive charts, size vs. distance distributions, and one-click CSV / JSON audit export.

---

## ??? Installation & Quick Start

### 1. Install Dependencies
Open PowerShell or Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Set Your Gemini API Key
You can set your API key as an environment variable or enter it directly in the app sidebar:
```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

### 3. Launch SubAquaVision
Run the one-click batch file:
```cmd
run.bat
```
Or directly via Streamlit:
```bash
streamlit run app.py
```

---

## ??? ROV Connection Guide

### ?? USB Connection (Direct Tether / Capture Card)
1. Plug your ROV USB camera or HDMI-to-USB capture card into your PC.
2. In the SubAquaVision sidebar, select **USB / UVC Camera**.
3. Select your camera index (e.g. `Camera Port 0` or `Camera Port 1`) and click **Connect ROV**.

### ?? Wi-Fi / RTSP / Network Connection
1. Ensure your PC is connected to the ROV's Wi-Fi network or Ethernet tether (e.g. `192.168.2.1`).
2. In the SubAquaVision sidebar, select **Wi-Fi (RTSP / HTTP Stream)**.
3. Enter the video URL:
   - **BlueROV2 / QGroundControl RTSP:** `rtsp://192.168.2.2:554/video`
   - **ESP32-CAM / MJPEG:** `http://192.168.1.100:81/stream`
   - **Raspberry Pi Camera Server:** `http://192.168.1.50:8080/video`
4. Click **Connect ROV**.

---

## ?? Optical Measurement Principle

Distance $D$ and real size $W_{real}$ are calculated using the pinhole lens model with Snell's law refraction:

$$D = \frac{f_{px} \cdot W_{real}}{W_{pixel}}$$

$$W_{real} = \frac{W_{pixel} \cdot D}{f_{px}}$$

Where $f_{px} = \frac{W_{image}}{2 \cdot \tan(\theta_{apparent} / 2)}$ and $\theta_{apparent} = 2 \cdot \arcsin\left(\frac{\sin(\theta_{air} / 2)}{1.333}\right)$.
