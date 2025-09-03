# 🖼️ Face Recognition System (with Face Detection)

A Python-based project using **OpenCV** and **LBPH Face Recognizer** for real-time **face detection and recognition** via webcam.  
This system can detect faces, recognize trained individuals, and flag unknown users.

---

## 🚀 Features
- 👤 **Face Detection** using Haar Cascade Classifier  
- 🧠 **Face Recognition** with LBPH Face Recognizer  
- 📂 **Dataset Support** – organize images per person in folders  
- 🎥 **Real-time Recognition** via webcam  
- ⚠️ **Unknown Face Handling** – flags and saves unknown snapshots  

---

## 🛠️ Tech Stack
- Python 3.x  
- OpenCV (cv2)  
- NumPy  

---

## 📂 Project Structure

Face-Recognition/
│── datasets/ # Training images (one folder per person)
│── haarcascade_frontalface_default.xml
│── face_recognition.py # Main script
│── input.jpg # Captured unknown faces
│── README.md

