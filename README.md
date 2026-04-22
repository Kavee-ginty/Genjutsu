# 🏆 Roboroarz 2025: Simulation Task

Welcome competitors! In this stage of the competition, your e-puck must autonomously navigate the Webots arena by detecting and decoding visual AR tags (ArUco markers) placed throughout the environment.

Computer vision is a core pillar of modern robotics. This document provides the exact technical specifications of the arena so that your vision algorithms can successfully interact with the simulated world.

## 📷 1. Arena Vision Specifications (CRITICAL)
If your computer vision pipeline is not configured to these exact parameters, your robot will be completely blind to the tags in the arena.

* **Target Dictionary:** `cv2.aruco.DICT_4X4_250`
  * *Note: The tags use a 4x4 grid and the IDs range from 0 to 249. If you initialize your detector with the wrong dictionary, it will return zero results.*
* **Marker Physical Dimensions:** `0.1 meters` (10 cm)
  * *Note: You will need this measurement if your team attempts 3D Pose Estimation (calculating distance and angle).*
* **Camera Resolution:** `640 x 480`
  * *Note: The default e-puck camera is too low-resolution to read tags from a distance. We have upgraded the arena e-pucks to 640x480. Ensure your Webots controller extracts the image at this resolution.*

## 🗺️ 2. Tag Data & Mapping
The ArUco tags encode simple integer IDs (e.g., `42`, `154`). 
To understand what these IDs mean for your mission objectives (e.g., target locations, obstacle warnings, or decimal multipliers), refer to the `arena_lookup_table.json` file included in this starter kit.Try to detect the Aruco tags with an atleast 0.3m distance clearance.

## 🛠️ 3. Recommended Development Milestones
You have three weeks to build a robust autonomous system. We highly recommend tackling the vision problem in the following order:

* **Milestone 1: Data Extraction.** Research how to enable the Webots `Camera` node and extract the raw byte array in your Python controller.
* **Milestone 2: The OpenCV Bridge.** Research how to convert a 1D raw byte array into a 3D NumPy matrix (BGR format) that the `cv2` library can understand.
* **Milestone 3: Detection.** Use the `cv2.aruco` module to find the tags in the image and print their integer IDs to your console.
* **Milestone 4: Spatial Awareness (Advanced).** Use the bounding box corners returned by the detector to calculate if the tag is to the left, right, or center of the robot, and program your wheel motors to react accordingly.

## ⚠️ 4. Setup Requirements
You are responsible for managing your own Python environments. Ensure your environment has the correct computer vision libraries installed before running the simulation:
```bash
pip install opencv-contrib-python numpy