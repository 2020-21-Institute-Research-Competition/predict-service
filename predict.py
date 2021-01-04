import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
import json


class Prediction:
    def __init__(self):
        super().__init__()
        self.__CLASS_INDEX = None
        self.__CLASS_INDEX_PATH = 'class_index.json'

    def __decode_predictions_modified(self, preds, top=1):
        if len(preds.shape) != 2 or preds.shape[1] != 3:
            raise ValueError(
                '`decode_predictions` expects ' 'a batch of predictions ''(i.e. a 2D array of shape (samples, 1000)). ' 'Found array with shape: ' + str(preds.shape))
        if self.__CLASS_INDEX is None:
            self.__CLASS_INDEX = json.load(open(self.__CLASS_INDEX_PATH))
        results = ''
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            for i in top_indices:
                # print(pred[i])
                results = self.__CLASS_INDEX[str(i)]
        return results

    def __prepare_image(self, file_name):
        img_path = 'images/predicted/'
        img = image.load_img(img_path + file_name, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    def predict(self, file_name, model):
        image = self.__prepare_image(file_name)
        predictions = model.predict(image)
        results = self.__decode_predictions_modified(predictions, top=1)
        return results
