from tensorflow.keras.models import load_model
import numpy as np
import cv2


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)

    def stream(self):
        model = load_model(r'training_models/apple_leaves_diseases_model.h5')
        class_labels = ['healthy', 'rust', 'scab']

        while True:
            ret, frame = self.__cap.read()

            cv2.imshow('frame', frame)
            print(type(frame)) #numpy array

            if not ret:
                print("Error: failed to capture image")
                break

            # TODO: Make a Prediction here

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #cv2.imwrite('frame.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')

        self.__cap.release()
        cv2.destroyAllWindows()

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
