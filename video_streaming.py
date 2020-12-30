import cv2
import os


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__file_name = 'frame.jpg'
        self.__cap = None
        if not os.path.exists('images'):
            os.makedirs('images')

    def stream(self):
        self.__cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            cv2.imwrite(f'images\\{self.__file_name}', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open(f'images\\{self.__file_name}', 'rb').read() + b'\r\n')

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
