import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
import json

CLASS_INDEX = None
CLASS_INDEX_PATH = 'class_index.json'


def decode_predictions_modified(preds, top=1):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 3:
        raise ValueError(
            '`decode_predictions` expects ' 'a batch of predictions ''(i.e. a 2D array of shape (samples, 1000)). ' 'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))
    results = ''
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            # print(pred[i])
            results = CLASS_INDEX[str(i)]
    return results


def prepare_image(file_name):
    img_path = 'images\\'
    img = image.load_img(img_path + file_name, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def predict(file_name, model):
    image = prepare_image(file_name)
    predictions = model.predict(image)
    results = decode_predictions_modified(predictions, top=1)
    print(results)
