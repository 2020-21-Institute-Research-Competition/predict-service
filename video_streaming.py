import cv2
import os
from tensorflow.keras.models import load_model
from predict import predict


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)
        if not os.path.exists('images'):
            os.makedirs('images')

    def stream(self):
        model = load_model(r'models/apple_leaves_diseases_model.h5')
        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # ? Save file in specific dir
                file_name = 'frame.jpg'
                cv2.imwrite(f'images\\{file_name}', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + open('images\\frame.jpg', 'rb').read() + b'\r\n')
                break
        self.__cap.release()
        cv2.destroyAllWindows()

        predict(file_name, model)

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
