import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Cache the model so it loads once
@st.cache_resource
def load_my_model(path="cifar10_cnn_model.h5"):
    model = load_model(path)
    return model

model = load_my_model("cifar10_cnn_model.h5")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classification (CNN)")
st.write("Upload an image (jpg/png). Model expects a tiny 32x32 RGB image — uploaded image will be resized automatically.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open image with PIL
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess: resize to 32x32, convert to array, scale
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_prob = preds[0][pred_idx]

    st.success(f"Predicted: **{class_names[pred_idx]}**  (probability: {pred_prob:.3f})")

    # Show full probability vector if user wants
    if st.checkbox("Show probabilities for all classes"):
        prob_map = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
        st.write(prob_map)
