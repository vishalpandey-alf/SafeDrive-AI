# 🚗 SafeDrive-AI: Real-Time Drowsiness Detection System

A real-time driver drowsiness detection tool using Python, OpenCV, and dlib. It monitors eye movements via facial landmarks and triggers an alert if the driver appears drowsy—enhancing road safety through AI.

---

## ⚙️ Features

- 👁️ Eye detection with dlib facial landmarks
- 📉 Calculates Eye Aspect Ratio (EAR) to detect drowsiness
- 🔊 Instant alarm using audio feedback
- 🧼 Clean modular structure for customization

---

## 🧠 Tech Stack

- Python
- OpenCV
- dlib
- NumPy, SciPy
- playsound/winsound

---

## 🚀 How to Run

1. Clone this repo  
   `git clone https://github.com/vishalpandey-alf/SafeDrive-AI.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Launch the system  
   `python main.py`

---

## 📁 Folder Structure

SafeDrive-AI/
├── main.py
├── utils/
├── alarm.wav
├── requirements.txt
└── README.md


##  Acknowledgment

This project is **inspired by the original drowsiness detector by [@shsarv](https://github.com/shsarv)**.  
We’ve customized and enhanced it for educational and portfolio purposes.

Credit to:
- @shsarv for the original idea
- OpenCV & dlib communities for foundational tools
