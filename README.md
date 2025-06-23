# Facial Recognition
 A real-time facial recognition web app using Streamlit and OpenCV. Supports face data collection, model training with LBPH, and live face recognition through your webcam.

# Facial Recognition System

A real-time facial recognition web app built using Streamlit and OpenCV. This project allows you to:

- Collect face data from webcam
- Train a face recognition model (LBPH)
- Perform live face recognition with confidence scores

Ideal for building prototypes of biometric attendance systems, access control applications, or learning facial recognition workflows.

## Features

- User-Friendly UI built with Streamlit
- Face Collection: Save 30 face samples per user with automatic detection
- Model Training using OpenCV's LBPH recognizer
- Real-Time Recognition with webcam feed
- Toggle Recognition with Start/Stop control

## Tech Stack

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Haar Cascade Classifier (face detection)
- LBPH Face Recognizer (face recognition)

## Installation

```bash
git clone https://github.com/yourusername/facial-recognition-app.git
cd facial-recognition-app
pip install -r requirements.txt
streamlit run app.py
```

## Directory Structure

```
facial-recognition-app/
├── faces/               # Collected face data per user
├── model.yml            # Trained face recognition model
├── labels.txt           # Mapping of label IDs to usernames
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md
```

## How to Use

1. Collect Faces: Enter a user ID and collect face samples using your webcam.
2. Train Model: Train the face recognition model using collected images.
3. Recognize Faces: Start real-time recognition. Known faces will be labeled with name and confidence.

## Notes

- Recognition works best under consistent lighting and angles.
- Confidence threshold is adjustable in code (default: <60 is considered a match).
- LBPH is suitable for small-scale applications; not ideal for large datasets.

## License

This project is licensed under the MIT License.

## Author

Developed by Nirmay Nandurkar.
