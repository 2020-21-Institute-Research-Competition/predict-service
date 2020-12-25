from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


class VideoStreaming:
    def __init__(self):
        super().__init__()
        self.__cap = cv2.VideoCapture(0)

    def stream(self):
        leaves_classifier = cv2.CascadeClassifier(
            r'classifiers/cascade.xml')
        model = load_model(r'training_models/apple_leaves_diseases_model.h5')

        while True:
            ret, frame = self.__cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            leaves_detected = leaves_classifier.detectMultiScale(gray, 1.32, 5)
            class_labels = ['healthy', 'rust', 'scab']

            print(leaves_detected)
            # print(type(frame))  # numpy array

            if not ret:
                print("Error: failed to capture image")
                break

            # TODO: Make a Prediction here
            for (x, y, w, h) in leaves_detected:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48),
                                      interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = image.img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi)[0]
                    label = class_labels[preds.argmax()]
                    label_position = (x, y)
                    cv2.putText(img, label, label_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'No Leaf Found', (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #cv2.imwrite('frame.jpg', frame)
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')

        self.__cap.release()
        cv2.destroyAllWindows()

    def close(self, signal_received, frame):
        self.__cap.release()
        cv2.destroyAllWindows()
