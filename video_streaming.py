import cv2
import os
from predict import predict

if not os.path.exists('images'):
    os.makedirs('images')


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)

    def stream(self):
        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ? Save file in specific dir
            file_name = 'frame.jpg'
            os.chdir('images')
            cv2.imwrite(file_name, frame)
            os.chdir('../')
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')
        predict(file_name)

        self.__cap.release()
        cv2.destroyAllWindows()

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
