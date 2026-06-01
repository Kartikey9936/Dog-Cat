import os

import streamlit as st
import requests
from PIL import Image
import io

# ── Page Config ────────────────────────────
st.set_page_config(
    page_title = "Dog vs Cat Classifier",
    page_icon  = "🐾",
    layout     = "centered"
)

# ── Header ─────────────────────────────────
st.title("🐾 Dog vs Cat Classifier")
st.markdown("Upload an image and it will predict if it's a **Dog** or **Cat**!")
st.divider()

# ── API URL ────────────────────────────────
# API_URL = "http://0.0.0.0:8000"
API_URL = os.getenv("API_URL", "http://localhost:8000")
# API_URL = "http://localhost:8000" # change after deploying backend

# ── Upload Image ───────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Show uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        # st.image(image, use_column_width=True)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Prediction")

        # Predict button
        if st.button("🔍 Classify", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Send to FastAPI
                    uploaded_file.seek(0)
                    files    = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded_file.name,
                               uploaded_file.getvalue(),
                               "image/jpeg")},
                        timeout=30
                    )

                    if response.status_code == 200:
                        data       = response.json()
                        prediction = data["prediction"]
                        confidence = data["confidence"]

                        # Show result
                        st.success(f"### {prediction}")
                        st.metric("Confidence", f"{confidence}%")

                        # Progress bar
                        st.progress(confidence / 100)

                        # Fun message
                        if "Dog" in prediction:
                            st.balloons()
                            st.write("🐶 Woof woof!")
                        else:
                            st.balloons()
                            st.write("🐱 Meow meow!")
                    else:
                        st.error(f"❌ API error {response.status_code}: {response.text}")

                except Exception as e:
                    st.error(f"❌ Cannot connect to backend: {e}")
                    st.info("Make sure FastAPI is running on port 8000")

# ── Footer ─────────────────────────────────
st.divider()
st.markdown(
    "<center>Built with ❤️ using Streamlit and Tea</center>",
    unsafe_allow_html=True
)