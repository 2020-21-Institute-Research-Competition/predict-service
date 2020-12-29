import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils


def prepare_image(file_name):
    img_path = 'images/'
    img = image.load_img(img_path + file_name, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def predict(file_name):
    model = load_model(r'models/apple_leaves_diseases_model.h5')
    image = prepare_image(file_name)
    predictions = model.predict(image)
    # * Get top 3 of predictions
    results = imagenet_utils.decode_predictions(predictions, top=3)
    print(results)
