import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

st.set_page_config(page_title="Real Mask Detection", layout="centered")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_model("model/mask-detector-model.h5")

st.title("Real Face Mask Detection")
st.write("Upload an image to detect if faces are wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    st.success(f"✅ Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        (mask, no_mask) = model.predict(face)[0]
        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(image, f"{label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Detection Result", use_container_width=True)

    pil_image = Image.fromarray(result_image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Result Image",
        data=byte_im,
        file_name="real_mask_detection_result.png",
        mime="image/png"
    )

    st.caption(f"🕒 Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.info("📷 Please upload an image to begin.")
