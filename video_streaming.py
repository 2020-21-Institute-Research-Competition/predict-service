from tensorflow.keras.models import load_model
import numpy as np
import cv2


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)

    def stream(self):
        model = load_model('training_models\\apple_leaves_diseases_model.h5')

        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            # TODO: Make a Prediction here

            cv2.imwrite('frame.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
