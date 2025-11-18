<div align="center">
  <h1>🧠 Head-Tracking Window Minimizer</h1>
  <p>A smart computer vision tool that minimizes your active window when you tilt your head down.</p>
  <br><br>

  ![Python](https://img.shields.io/badge/Python-3.10+-blue)
  ![OpenCV](https://img.shields.io/badge/OpenCV-CV2-orange)
  ![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-green)
  ![Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Mac%20%7C%20Linux-purple)
</div>

## 🚀 Overview
**Head-Tracking Window Minimizer** uses your webcam to detect head pitch angle using MediaPipe’s FaceMesh.  
When you look *down* for a certain amount of time (default: **0.3 seconds**), the tool automatically **minimizes your active window**.

Perfect for:
- Students secretly watching series in class 😂  
- Quick minimizing during work  
- Privacy-conscious users  
- Productivity setups


 <img width="1280" height="640" alt="Discord bots (1)" src="https://github.com/user-attachments/assets/45b9410c-032a-444d-a42f-18fdb0b7d5a0" />

## Demo

https://youtu.be/6Go971NKeHk

## ✨ Features

### 🎯 **Smart Head Tracking**
- Uses MediaPipe FaceMesh for accurate head pose
- Calculates **pitch angle** in real time  
- Smooth motion using moving average

### 🖥️ **Window Control**
- Auto minimize currently active window  
- Works on **Windows, macOS, Linux**

### ⚙️ **Filter Modes**
You can choose to:
- 🟢 **Minimize ALL windows**
- 🟡 **Whitelist only specific apps**
- 🔴 **Blacklist apps you never want minimized**

### 🎛️ Real-time UI Overlay
- Live pitch angle  
- Status display  
- FPS counter  
- Circular angle gauge  
- Total minimize count  

### ⌨️ Keyboard Controls
| Key | Action |
|-----|--------|
| **Q** | Quit program |
| **P** | Pause / Resume tracking |
| **C** | Calibrate neutral head position |
| **+ / -** | Increase or decrease sensitivity |



## 📦 Requirements
This project is tested and works **only on Python 3.11**.

Make sure you install exactly:

- **Python 3.11.x** (recommended: 3.11)
- Webcam
- OS Support:
  - Windows ✔
  - macOS ✔
  - Linux ✔

Need a webcam to monitor
### Required Python Packages
All dependencies compatible with Python 3.11:



## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/headtrack-minimizer
cd headtrack-minimizer
pip install opencv-python mediapipe numpy pygetwindow pyautogui
python -c "import cv2; import mediapipe; import pygetwindow; print('All packages installed successfully!')"
headtrack_env\Scripts\activate
python -u headtrack.py
```

### `LICENSE`
```
MIT License

Copyright (c) 2025 QuoRith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
