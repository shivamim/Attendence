import os
import face_recognition
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ============================
# 1. Initialize Directories
# ============================
TRAINING_DIR = "Training_Images"
ATTENDANCE_DIR = "Attendance"

if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)

if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# ============================
# 2. Encode Training Images
# ============================
def train_faces(training_dir):
    """
    Encodes all faces from images inside Training_Images.
    """
    encodings = []
    names = []
    
    for person_name in os.listdir(training_dir):
        person_folder = os.path.join(training_dir, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings.append(face_encoding)
                    names.append(person_name)
    return encodings, names

# ============================
# 3. Save Attendance
# ============================
def mark_attendance(name):
    """
    Marks attendance in a CSV file based on today's date.
    """
    today = datetime.now().strftime("%d-%m-%Y")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name", "Timestamp"])

    # Check if attendance already marked
    if name not in df["Name"].values:
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_entry = {"Name": name, "Timestamp": timestamp}
        df = df.append(new_entry, ignore_index=True)
        df.to_csv(file_path, index=False)
        st.success(f"✅ Attendance marked for {name}")
    else:
        st.info(f"ℹ️ Attendance already marked for {name}")

# ============================
# 4. Recognize Faces
# ============================
def recognize_faces(known_encodings, known_names, captured_image):
    """
    Recognizes faces from a captured image and returns matching names.
    """
    recognized_names = []
    image = face_recognition.load_image_file(captured_image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        recognized_names.append(name)
    return recognized_names

# ============================
# 5. Streamlit UI
# ============================
st.title("📷 Face Recognition Attendance System")

# Step 1: Train Model
st.header("Step 1: Train Faces")
if st.button("Train Faces"):
    known_encodings, known_names = train_faces(TRAINING_DIR)
    np.save("known_encodings.npy", known_encodings)
    np.save("known_names.npy", known_names)
    st.success("🎉 Training completed successfully!")

# Step 2: Capture and Recognize
st.header("Step 2: Mark Attendance")

uploaded_file = st.file_uploader("Upload a captured image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load trained model
    if os.path.exists("known_encodings.npy") and os.path.exists("known_names.npy"):
        known_encodings = np.load("known_encodings.npy", allow_pickle=True)
        known_names = np.load("known_names.npy", allow_pickle=True)
        
        # Save uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Recognize faces
        recognized_names = recognize_faces(known_encodings, known_names, temp_image_path)
        if recognized_names:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("### Recognized Faces:")
            for name in recognized_names:
                if name != "Unknown":
                    st.success(f"✔️ {name}")
                    mark_attendance(name)
                else:
                    st.warning("❗ Unknown Face Detected")
        else:
            st.error("No faces detected in the uploaded image.")
    else:
        st.error("⚠️ Please train the model first using 'Train Faces'.")

# Step 3: View Attendance
st.header("Step 3: View Attendance")

today = datetime.now().strftime("%d-%m-%Y")
file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.write("### Attendance for Today:")
    st.dataframe(df)
else:
    st.info("⚠️ No attendance has been marked for today.")
