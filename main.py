import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Facial Recognition App", layout="centered")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def collect_faces():
    user_id = st.text_input("Enter user ID:")
    start = st.button("Start Collecting Faces")

    if user_id and start:
        os.makedirs(f'faces/{user_id}', exist_ok=True)
        cap = cv2.VideoCapture(0)
        count = 0
        stframe = st.empty()

        while cap.isOpened() and count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            faces = face_classifier.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f'faces/{user_id}/face_{count}.jpg', face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        st.success(f"Captured {count} faces for {user_id}")


def train_model():
    data_path = 'faces/'
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(data_path):
        st.warning("No face data found.")
        return

    for user_id in os.listdir(data_path):
        user_folder = os.path.join(data_path, user_id)
        if not os.path.isdir(user_folder):
            continue

        label_map[current_label] = user_id

        for filename in sorted(os.listdir(user_folder)):
            img_path = os.path.join(user_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(current_label)
        current_label += 1

    if not faces:
        st.warning("No face images found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save('model.yml')

    with open("labels.txt", "w") as f:
        for label_id, user_id in label_map.items():
            f.write(f"{label_id},{user_id}\n")

    st.success("Model trained and saved as 'model.yml'")


def recognize_faces():
    if not os.path.exists("model.yml") or not os.path.exists("labels.txt"):
        st.warning("Model or labels not found. Train the model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model.yml')

    label_map = {}
    with open("labels.txt", "r") as f:
        for line in f:
            label_id, user_id = line.strip().split(",")
            label_map[int(label_id)] = user_id

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop = st.button("Stop Recognition")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        faces = face_classifier.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)

            if confidence < 60:
                user_id = label_map.get(label, "Unknown")
                text = f"{user_id} ({confidence:.1f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()



st.title("Facial Recognition System")
option = st.sidebar.radio("Select Action", ["Collect Faces", "Train Model", "Recognize Faces"])

if option == "Collect Faces":
    collect_faces()
elif option == "Train Model":
    train_model()
elif option == "Recognize Faces":
    recognize_faces()
