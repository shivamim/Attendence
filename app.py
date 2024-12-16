import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Streamlit UI Title
st.title("📋 Attendance Tracker")

# Get today's date in the required format
date = datetime.now().strftime("%d-%m-%Y")

# Define the path to the attendance file
file_path = f"Attendance/Attendance_{date}.csv"

# Check if the 'Attendance' folder exists; if not, create it
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Function to display attendance data
def display_attendance():
    if os.path.exists(file_path):
        # If the file exists, read and display it
        df = pd.read_csv(file_path)
        st.success(f"✅ Attendance data for {date}")
        st.dataframe(df.style.highlight_max(axis=0))
    else:
        # If file does not exist, handle gracefully
        st.warning(f"⚠️ Attendance file for {date} does not exist.")
        # Create an empty DataFrame with relevant columns
        columns = ["Name", "Status", "Timestamp"]  # Adjust columns as needed
        df = pd.DataFrame(columns=columns)
        st.info("Creating an empty attendance sheet.")
        st.dataframe(df)

# Streamlit UI Display
st.header(f"Attendance for {date}")
display_attendance()

# Optional: Allow user to add a new attendance entry
st.subheader("Add New Entry")
name = st.text_input("Enter Name:")
status = st.selectbox("Status", ["Present", "Absent", "Late"])
timestamp = datetime.now().strftime("%H:%M:%S")

if st.button("Add Entry"):
    # Append the new entry to the file
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Create an empty DataFrame if the file does not exist
        df = pd.DataFrame(columns=["Name", "Status", "Timestamp"])
    
    # Add the new entry
    new_entry = {"Name": name, "Status": status, "Timestamp": timestamp}
    df = df.append(new_entry, ignore_index=True)
    df.to_csv(file_path, index=False)
    st.success("✅ Entry added successfully!")
    st.experimental_rerun()  # Refresh the app to show updated data
