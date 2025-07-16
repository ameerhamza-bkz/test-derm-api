import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os

token = st.secrets.get("HF_TOKEN")

# Initialize Gradio client
client = Client("ameerhmz/derm-foundation", hf_token = token)

st.title("Skin Condition Predictor")

st.markdown(
    """
    Upload an image of the skin or provide an image URL to predict the skin condition using the Derm Foundation model.
    """
)

# Input option: Upload or URL
input_option = st.radio("Select input type:", ["Upload Image", "Image URL"])

image = None
image_source = None

if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            image_source = tmp_file.name

elif input_option == "Image URL":
    url = st.text_input("Enter image URL:")
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)
            image_source = url
        except Exception as e:
            st.error(f"Failed to load image from URL: {e}")

if image_source:
    if st.button("Predict Skin Condition"):
        with st.spinner("Predicting..."):
            try:
                result = client.predict(
                    image=handle_file(image_source),
                    api_name="/predict_skin_condition"
                )
                st.success("Prediction complete!")
                st.json(result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
