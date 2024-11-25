import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO(r"C:\Users\Varam\Downloads\best(3).pt")

# Define the Streamlit app
def main():
    st.title("Solar Panel Defect Detection")
    st.write("Upload an image to detect defects")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect'):
            # Perform inference on the uploaded image
            results = model(image)

            # Extract image with bounding boxes and labels
            result_img = results[0].plot()  # Plotting the first result (assuming only one image is processed at a time)

            # Convert the result_img to a format that can be displayed with st.image
            result_img_pil = Image.fromarray(result_img)
            st.image(result_img_pil, caption='Detection Result', use_column_width=True)

            # Displaying details of each detection
            for detection in results[0].boxes:
                cls_id = int(detection.cls)  # Class ID
                score = float(detection.conf)  # Confidence score
                bbox = detection.xyxy.numpy().astype(int).tolist()  # Bounding box coordinates
                st.write(f"Predicted class: {results.names[cls_id]} with confidence {score:.2f}")
                st.write(f"Bounding box: {bbox}")

# Run the app
if __name__ == "__main__":
    main()
