import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st
from PIL import Image
import io
import tempfile


def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)


def predict(model, img1, img2, target_size=(224, 224)):
    try:
        # Preprocess image 1
        img1 = img1.resize(target_size).convert("L")  # Convert to grayscale
        img1 = np.array(img1) / 255.0  # Normalize
        img1 = np.expand_dims(img1, axis=(0, -1))  # Add batch and channel dimensions

        # Preprocess image 2
        img2 = img2.resize(target_size).convert("L")  # Convert to grayscale
        img2 = np.array(img2) / 255.0  # Normalize
        img2 = np.expand_dims(img2, axis=(0, -1))  # Add batch and channel dimensions

        # Predict similarity
        prediction = model.predict([img1, img2])

        # Interpret the output
        similarity = prediction[0][0]
        return 1 if similarity > 0.5 else 0  # 1 = Dissimilar, 0 = Similar
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def main():
    st.title("Image Matcher")

    # File upload for the model
    siamese = st.file_uploader("Upload your Siamese model (.h5 format)", type=["h5"])

    siamese_model = None
    if siamese:
        try:
            # Save the uploaded model temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
                temp_model_file.write(siamese.read())
                temp_model_file_path = temp_model_file.name

            # Load the model
            siamese_model = load_model(
                temp_model_file_path,
                custom_objects={"cosine_distance": cosine_distance}
            )
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # File upload for images
    user_input1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    user_input2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    if siamese_model and user_input1 and user_input2:
        img1 = Image.open(io.BytesIO(user_input1.read()))
        img2 = Image.open(io.BytesIO(user_input2.read()))

        st.image(img1, caption="Image 1", use_column_width=True)
        st.image(img2, caption="Image 2", use_column_width=True)

        if st.button("Process"):
            predicted_label = predict(siamese_model, img1, img2)
            if predicted_label is not None:
                st.write(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
            else:
                st.error("An error occurred during prediction.")
    else:
        st.info("Please upload both the model and images to proceed.")


if __name__ == "__main__":
    main()
