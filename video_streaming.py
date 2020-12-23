import cv2


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)

    def stream(self):
        healthCascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")

        while True:
            ret, frame = self.__cap.read()

            if not ret:
                print("Error: failed to capture image")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            healths = healthCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in healths:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imwrite('demo.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
