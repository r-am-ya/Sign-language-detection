# ✋ Sign Language Detection using Deep Learning

## 📌 Project Overview
This project implements a real-time American Sign Language (ASL) recognition system using MediaPipe hand landmarks and a Deep Learning model. The system captures hand gestures through a webcam, extracts landmark features, and predicts the corresponding ASL alphabet (A–Z) in real time using a trained neural network.

---

## 🎯 Objectives
- Collect ASL hand landmark data using MediaPipe  
- Train a deep learning model for ASL alphabet classification  
- Develop a real-time GUI-based ASL recognition system  
- Enable offline, real-time gesture recognition  

---

## 🛠️ Technologies Used
- Python  
- MediaPipe  
- TensorFlow / Keras  
- OpenCV  
- Tkinter  
- NumPy  
- Scikit-learn  

---

## 📂 Project Structure

Sign-language-detection/

├── COLLECT.py # Data collection script

├── TRAINING.py # Model training script

├── GUI.py # Real-time ASL recognition GUI

├── asl_landmark_dl_model.h5 # Trained model

├── README.md


---

## 🧠 Project Workflow

### 1️⃣ Data Collection
- MediaPipe is used to detect hand landmarks
- Landmark coordinates are saved for each ASL alphabet

### 2️⃣ Model Training
- Landmarks are normalized relative to wrist position
- A deep learning model is trained to classify A–Z gestures
- The trained model is saved as a `.h5` file

### 3️⃣ Real-Time Prediction
- Webcam input is processed in real time
- Hand landmarks are extracted and normalized
- The predicted ASL alphabet and confidence score are displayed in the GUI

📌 Future Enhancements

- Word and sentence-level ASL recognition
- Prediction smoothing for stable outputs
- Web or mobile-based deployment
