import cv2
import os
import datetime
import requests
import json

from capture import Capture


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__file_name = 'frame.jpg'
        self.__cap = cv2.VideoCapture(0)
        if not os.path.exists('images'):
            os.makedirs('images')

    def stream(self):
        start_time = datetime.datetime.now()
        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            if (datetime.datetime.now() - start_time).total_seconds() >= 10:
                capture = Capture(frame)
                try:
                    headers = {'Content-Type': 'application/json'}
                    predicted_data = json.dumps(capture.run())
                    res = requests.post(
                        'http://192.168.180.109:8080/api/v1/CZon9KvrGpNouoX0rwXi/telemetry', headers=headers, data=predicted_data)
                    print(res)
                except Exception as ex:
                    print(ex)
                    continue
                start_time = datetime.datetime.now()

            cv2.imwrite(f'images/{self.__file_name}', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open(f'images/{self.__file_name}', 'rb').read() + b'\r\n')

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
