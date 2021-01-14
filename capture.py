import csv
from tensorflow.keras.models import load_model

from predict import Prediction


class Capture:
    def __init__(self, ID, filename, path):
        super().__init__()
        self.__prediction = Prediction()
        self.__model = load_model(r'models/apple_leaves_diseases_model.h5')
        self.__ID = ID
        self.__path = path
        self.__filename = filename

    def run(self):
        with open('predicted_results.csv', 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            result = self.__prediction.predict(self.__filename, self.__model)
            csv_writer.writerow([self.__ID, result, self.__path])
            return {'id': str(self.__ID), 'status': result, 'path': self.__path}
