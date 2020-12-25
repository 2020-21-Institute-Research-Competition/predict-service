from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

leaves_classifier = cv2.CascadeClassifier(
    r'classifiers/cascade.xml')
model = load_model(r'training_models/apple_leaves_diseases_model.h5')

filename = r'test/Test_301.jpg'
img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
leaves_detected = leaves_classifier.detectMultiScale(gray, 1.32, 5)
class_labels = ['healthy', 'rust', 'scab']
print(leaves_detected)

for (x, y, w, h) in leaves_detected:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

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

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
