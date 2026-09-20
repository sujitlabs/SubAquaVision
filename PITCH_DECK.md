# ?? SubAquaVision: Hackathon Presentation & Pitch Deck Guide

## ?? Project Tagline
> **"Turning any ROV camera into an intelligent subsea surveyor with zero-shot AI, optical metrology, and real-time hazard tracking."**

---

## ?? 1. The Problem Statement
Underwater exploration, marine conservation, and subsea infrastructure inspections face 4 massive hurdles:
1. **Severe Optical Distortion:** Seawater absorbs red light within 3-5 meters, leaving images hazy, greenish-blue, and low-contrast.
2. **Costly Hardware:** Commercial ROVs require expensive laser scaling systems and stereo cameras ($5,000+) to measure fish or pipeline cracks.
3. **Data Overload & Double Counting:** Video surveyors spend hours manually re-counting marine life or ocean plastics.
4. **Lack of Interoperability:** Most vision software is locked to proprietary hardware and cannot easily plug into USB cameras, BlueROV2, or custom AUVs.

---

## ?? 2. Our Solution: SubAquaVision
SubAquaVision is a **100% pure-software subsea vision intelligence platform** that connects to any camera (USB, Wi-Fi, RTSP) and delivers real-time detection, measurement, and reporting.

### ?? Core Capabilities:
1. **Gemini 2.5 Flash Multimodal Vision:**
   - Zero-shot spatial detection with normalized 2D bounding boxes.
   - Multi-class categorization: Marine Life, Plastic Debris, Subsea Pipelines, Divers.
   - Taxonomic classification with scientific naming & IUCN conservation/hazard status.
2. **Optical Size & Distance Metrology (Laserless):**
   - Perspective pinhole geometry with Snell's law seawater refraction correction ($n = 1.333$).
   - Calculates real-world dimensions ($L \times W$ in cm/m) and distance from camera ($D$ in meters).
3. **Underwater Color Restoration:**
   - Real-time Red-channel attenuation compensation and LAB-space CLAHE to eliminate turbidity.
4. **Persistent Multi-Object Tracking (MOT):**
   - IoU-based tracking assigns persistent IDs (`#Track 1`, `#Track 2`), preventing duplicate counting in video feeds.
5. **Military/Scientific ROV Telemetry HUD:**
   - Live overlay showing Depth, Heading Compass, Water Pressure, and Temperature.
6. **One-Click Mission Audit Reports:**
   - Generates official, print-ready PDF/HTML survey logs with detection tables and stats.

---

## ?? 3. 3-Minute Live Demo Script for Hackathon Judges

| Time | Action | What to Say |
| :--- | :--- | :--- |
| **0:00 - 0:30** | Open Dashboard (`localhost:8501`) | *"Judges, underwater environments are challenging due to color distortion and turbidity. SubAquaVision turns standard ROV video into high-accuracy actionable intelligence."* |
| **0:30 - 1:00** | Click **Analyze Sample 1** (Coral Reef) | *"Notice how our OpenCV CLAHE pipeline restores the missing red spectrum. Gemini 2.5 Flash detects the sea turtle and coral with exact bounding boxes, scientific names, and health status."* |
| **1:00 - 1:40** | Highlight **Measurement & HUD** | *"Notice the size and distance tag: 92?64 cm at 2.2 meters. We use optical refraction physics ($n=1.333$) to calculate dimensions without expensive hardware laser scalers."* |
| **1:40 - 2:20** | Show **Debris & Tracking (Tab 4)** | *"When marine plastics or ghost nets are spotted, the system triggers a critical hazard alert. Our Multi-Object Tracker ensures unique counting without duplicates."* |
| **2:20 - 3:00** | Click **Download Mission Audit Report** | *"Finally, with 1 click, the operator gets a printable, official mission audit report ready for marine biologists or offshore engineers."* |

---

## ? 4. Anticipated Judge Questions & Winning Answers

**Q: Why use Gemini Flash instead of a custom-trained YOLO model?**  
> *"Custom YOLO models require thousands of labeled underwater images and struggle on unseen species. Gemini Flash provides zero-shot spatial bounding boxes, understands thousands of rare marine species, evaluates habitat health, and answers natural language questions zero-shot."*

**Q: How accurate is your laserless distance and size measurement?**  
> *"By combining the camera's field of view, focal length, and the optical refraction index of seawater ($n = 1.333$), our pinhole geometry model estimates physical dimensions within an acceptable margin for field surveys without needing a \$5,000 laser scaler."*

**Q: Can this connect to any ROV hardware?**  
> *"Yes! We built a universal stream reader supporting direct USB/UVC cameras, BlueROV2 RTSP streams, ESP32-CAMs, and standard subsea IP cameras."*
