import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import os
import streamlit as st
from PIL import Image
import io
import tempfile


def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)

def predict(model, image_pair, target_size=(224, 224)):
    try:
        img1_path, img2_path = image_pair

        # Verify files exist
        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"File not found: {img1_path}")
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"File not found: {img2_path}")

        # Load and preprocess images
        img1 = load_img(img1_path, target_size=target_size, color_mode='grayscale')
        img1 = img_to_array(img1) / 255.0
        img1 = np.expand_dims(img1, axis=0)  # Add batch dimension

        img2 = load_img(img2_path, target_size=target_size, color_mode='grayscale')
        img2 = img_to_array(img2) / 255.0
        img2 = np.expand_dims(img2, axis=0)  # Add batch dimension

        # Predict the similarity of the pair
        prediction = model.predict([img1, img2])

        # Interpret the model output
        similarity = prediction[0][0]
        return 1 if similarity > 0.5 else 0  # 1 for dissimilar, 0 for similar

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    st.title("Image Matcher")
    siamese = st.file_uploader('Upload your model (.h5 format)', type=["h5"])

    siamese_model = None
    if siamese:
        try:
            # Save the uploaded model file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
                temp_model_file.write(siamese.read())
                temp_model_file_path = temp_model_file.name

            # Load the model
            siamese_model = load_model(
                temp_model_file_path,
                custom_objects={'cosine_distance': cosine_distance}
            )
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return


    # Upload images
    user_input1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    user_input2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    if user_input1 and user_input2:
        # Convert uploaded files to PIL images
        img1 = Image.open(io.BytesIO(user_input1.read()))
        img2 = Image.open(io.BytesIO(user_input2.read()))

        # Display uploaded images
        st.image(img1, caption="Image 1", use_column_width=True)
        st.image(img2, caption="Image 2", use_column_width=True)

        st.button('Process')

        # Make a prediction
        predicted_label = predict(siamese_model, img1, img2)

        if predicted_label is not None:
            st.write(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
        else:
            st.error("An error occurred during prediction.")
    else:
        st.info("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
