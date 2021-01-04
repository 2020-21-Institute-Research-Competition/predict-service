import cv2
import datetime
import threading
import csv
import uuid
from predict import Prediction
from tensorflow.keras.models import load_model


class Capture(threading.Thread):
    def __init__(self):
        super().__init__()
        self.__prediction = Prediction()
        self.__model = load_model(r'models/apple_leaves_diseases_model.h5')
        self.__model.make_predict_function()

    def run(self):
        start_time = datetime.datetime.now()
        while True:
            # Change to default seconds later
            if (datetime.datetime.now() - start_time).total_seconds() >= 5:
                ID = uuid.uuid1()
                file_name = f'{ID}.jpg'  # Change to id later
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()

                if not ret:
                    print("Error: failed to capture image")
                    break

                cv2.imwrite(f'images/predicted/{file_name}', frame)
                with open('predicted_results.csv', 'a', newline='', encoding='utf-8') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    result = self.__prediction.predict(file_name, self.__model)
                    csv_writer.writerow(
                        [ID, result, f'images/predicted/{file_name}'])

                cap.release()
                cv2.destroyAllWindows()
                start_time = datetime.datetime.now()
