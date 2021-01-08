import uuid
import csv
import cv2
from tensorflow.keras.models import load_model

from predict import Prediction


class Capture:
    def __init__(self, frame):
        super().__init__()
        self.__prediction = Prediction()
        self.__model = load_model(r'models/apple_leaves_diseases_model.h5')
        self.__frame = frame

    def run(self):
        ID = uuid.uuid1()
        file_name = f'{ID}.jpg'
        image_path = f'images/predicted/{file_name}'

        cv2.imwrite(image_path, self.__frame)
        with open('predicted_results.csv', 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            result = self.__prediction.predict(file_name, self.__model)
            csv_writer.writerow([ID, result, image_path])
            return {'id': str(ID), 'status': result, 'path': image_path}
