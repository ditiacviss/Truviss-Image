import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st
from PIL import Image
import io


# Custom loss function for the Siamese model
def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)


# Prediction function
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


# Main Streamlit application
def main():
    st.title("Image Matcher")

    # Load the pre-trained Siamese model
    model_path = "siamese_model.h5"  # Ensure this file exists in the working directory
    siamese_model = None

    try:
        siamese_model = load_model(
            model_path,
            custom_objects={"cosine_distance": cosine_distance}
        )
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File upload for images
    user_input1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    user_input2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    if user_input1 and user_input2:
        # Convert uploaded files to PIL images
        img1 = Image.open(io.BytesIO(user_input1.read()))
        img2 = Image.open(io.BytesIO(user_input2.read()))

        # Display uploaded images
        st.image(img1, caption="Image 1", use_column_width=True)
        st.image(img2, caption="Image 2", use_column_width=True)

        # Process button
        if st.button("Process"):
            predicted_label = predict(siamese_model, img1, img2)
            if predicted_label is not None:
                st.write(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
            else:
                st.error("An error occurred during prediction.")
    else:
        st.info("Please upload both images to proceed.")


if __name__ == "__main__":
    main()
