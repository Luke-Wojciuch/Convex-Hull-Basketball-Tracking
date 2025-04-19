# 🏀 Convex-Hull-Basketball-Tracking

**A real-time computer vision system that analyzes geometric formations in basketball by tracking players, referees, and the ball using YOLOv8 and ByteTrack.** The system maps player positions onto a radar-style minimap, highlighting spatial arrangements, defensive structures, and offensive sets as they evolve during live gameplay.

---

## 📖 Table of Contents

1. [📌 Project Description]  
2. [🎥 Demo]
3. [🛠 Useability]
4. [⚙️ Technologies Used]
5. [👏 Credits]

---

## 📌 Project Description

### What It Does  
This system processes basketball game footage in real time to track players, referees, and the ball. It then maps this data onto a radar-style minimap, providing a geometric analysis of team formations. The radar can be displayed in the corner of the screen or expanded to full view, offering coaches, analysts, or fans a dynamic understanding of team structure and spacing.

### Why These Technologies  
- **YOLOv8** for real-time, high-accuracy object detection.  
- **ByteTrack** for robust and stable multi-object tracking.  
- **OpenCV** for efficient frame processing and visual rendering.  

### Challenges Faced  
- Ensuring **accurate tracking** in high-speed or congested gameplay environments.  
- Implementing a reliable **homography transformation** to project positions from video footage to a bird's-eye radar view, adaptable across camera angles.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/rS2Ayo4zWac/0.jpg)](https://youtu.be/rS2Ayo4zWac)

> Click the image above or [watch here on YouTube](https://youtu.be/rS2Ayo4zWac)

---

## 🛠 Useability

Clone the repository and install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/basketball-vision-tracker.git
cd basketball-vision-tracker

# (Optional) Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## ⚙️ Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics) – Fast and accurate object detection.
- [ByteTrack](https://github.com/ifzhang/ByteTrack) – Robust multi-object tracking algorithm.
- [OpenCV](https://opencv.org/) – Real-time computer vision library used for frame processing and rendering overlays.
- [NumPy](https://numpy.org/) – Numerical operations on arrays and coordinates.
- [Matplotlib](https://matplotlib.org/) – Optional visualizations and plotting.
- [cvzone](https://github.com/cvzone/cvzone) – Simplifies OpenCV tasks for easier video pipeline handling.
- Python – The core language used for stitching all components together.

---

## 👏 Credits

- Detection powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- Tracking system based on [ByteTrack](https://github.com/ifzhang/ByteTrack)  
- Radar and homography techniques inspired by:
  - [Radar View for Soccer with OpenCV – Medium](https://medium.com/@ibrahimokdadov/soccer-tracking-and-radar-visualization-in-python-400ef8786121)  
  - [Basketball Homography with OpenCV – Roboflow](https://medium.com/@roboflow/basketball-player-tracking-and-homography-with-opencv-125872b538a0)  
  - [YOLOv5 + DeepSORT Multi-Object Tracker – GitHub](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

