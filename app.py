import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Set paths
FACES_DIR = "known_faces"
ATTENDANCE_DIR = "Attendance"
ENCODINGS_FILE = "face_encodings.csv"

# Create directories if they don't exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Function to load encodings from CSV
def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        return pd.read_csv(ENCODINGS_FILE)
    else:
        return pd.DataFrame(columns=["Name", "Encoding"])

# Function to save encodings to CSV
def save_encodings(df):
    df.to_csv(ENCODINGS_FILE, index=False)

# Function to encode face and add to dataset
def add_face(name, image_file):
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        encoding = encodings[0]
        df_encodings = load_encodings()
        encoding_str = np.array2string(encoding, separator=",", max_line_width=np.inf)
        df_encodings = df_encodings.append({"Name": name, "Encoding": encoding_str}, ignore_index=True)
        save_encodings(df_encodings)
        return "Face added successfully!"
    else:
        return "No face detected in the image!"

# Function to compare face and mark attendance
def mark_attendance(test_image_file):
    test_image = face_recognition.load_image_file(test_image_file)
    test_encoding = face_recognition.face_encodings(test_image)

    if len(test_encoding) == 0:
        return "No face detected in the uploaded image."

    test_encoding = test_encoding[0]
    df_encodings = load_encodings()

    known_encodings = []
    known_names = []

    for _, row in df_encodings.iterrows():
        known_names.append(row["Name"])
        known_encodings.append(np.fromstring(row["Encoding"][1:-1], sep=","))

    # Compare face encodings
    matches = face_recognition.compare_faces(known_encodings, test_encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_names[match_index]
        mark_attendance_csv(name)
        return f"Attendance marked for {name}"
    else:
        return "No match found. Face not recognized."

# Function to mark attendance in CSV
def mark_attendance_csv(name):
    date = datetime.now().strftime("%d-%m-%Y")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Time"])
    else:
        df = pd.read_csv(file_path)

    time_now = datetime.now().strftime("%H:%M:%S")
    df = df.append({"Name": name, "Time": time_now}, ignore_index=True)
    df.to_csv(file_path, index=False)

# Streamlit UI
st.title("Face Recognition Attendance System")

menu = ["Add Face", "Mark Attendance"]
choice = st.sidebar.selectbox("Select an Option", menu)

if choice == "Add Face":
    st.header("Add a New Face")
    name = st.text_input("Enter the name of the person:")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Add Face"):
        if name and image_file:
            result = add_face(name, image_file)
            st.success(result)
        else:
            st.warning("Please provide a name and upload an image.")

elif choice == "Mark Attendance":
    st.header("Mark Attendance")
    test_image_file = st.file_uploader("Upload an image to mark attendance", type=["jpg", "jpeg", "png"])

    if st.button("Mark Attendance"):
        if test_image_file:
            result = mark_attendance(test_image_file)
            st.success(result)
        else:
            st.warning("Please upload an image.")
