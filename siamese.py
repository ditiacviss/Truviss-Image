import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def predict(model, image_pair, image_folder, target_size=(224, 224)):

    try:
        img1_path, img2_path = image_pair
        img1_full_path = os.path.join(image_folder, img1_path)
        img2_full_path = os.path.join(image_folder, img2_path)

        # Verify files existkkk
        if not os.path.exists(img1_full_path):
            raise FileNotFoundError(f"File not found: {img1_full_path}")
        if not os.path.exists(img2_full_path):
            raise FileNotFoundError(f"File not found: {img2_full_path}")

        # Load images
        img1 = load_img(img1_full_path, target_size=target_size)
        img1 = img_to_array(img1) / 255.0

        img2 = load_img(img2_full_path, target_size=target_size)
        img2 = img_to_array(img2) / 255.0

        # Predict the similarity of the pair
        prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

        # The model outputs a probability (0 or 1)
        similarity = prediction[0][0]
        return 1 if similarity > 0.5 else 0  # Similar if probability > 0.5, otherwise dissimilar

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

siamese_model = load_model('siamese.h5')
image_folder = 'LOGOS/data-sm/all_images'  # Ensure this matches the path to your images
image_pair = ('LOGOS/data-sm/all_images/2016-super-bowl-vector-logo-400x400.jpg', 'LOGOS/data-sm/all_images/noisy_image-2016-super-bowl-vector-logo-400x400.jpg')  # Example image pair
predicted_label = predict(siamese_model, image_pair, image_folder)

if predicted_label is not None:
    print(f"Prediction: {'Similar' if predicted_label == 0 else 'Dissimilar'}")
