import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from tensorflow.keras import backend as K

input_img1= input('Path for Img1: ')
input_img2= input('Path for Img2: ')

def predict(model, image_pair, target_size=(224, 224)):

    try:
        img1_path, img2_path = image_pair

        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"File not found: {img1_path}")
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"File not found: {img2_path}")

        # Load images
        img1 = load_img(img1_path, target_size=target_size, color_mode='grayscale')
        img1 = img_to_array(img1) / 255.0

        img2 = load_img(img2_path, target_size=target_size, color_mode='grayscale')
        img2 = img_to_array(img2) / 255.0

        prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

        similarity = prediction[0][0]
        print(f'Similarity Score: {similarity}')
        return 1 if similarity > 0.5 else 0

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)

siamese_model = load_model(r"siamese_model.h5",custom_objects={'cosine_distance': cosine_distance})
image_pair = (input_img1, input_img2)
predicted_label = predict(siamese_model, image_pair)

if predicted_label is not None:
    print(f"Prediction: {'Similar' if predicted_label == 0 else 'Dissimilar'}")
