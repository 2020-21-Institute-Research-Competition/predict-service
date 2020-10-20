from time import time, sleep
from datetime import datetime, timedelta
import cv2
from pathlib import Path

Path("./images").mkdir(parents=True, exist_ok=True)


def imageCapturer(image_name, wait_time):
    cap = cv2.VideoCapture(0)
    ts = int(time())
    finish_time = datetime.now() + timedelta(seconds=wait_time)

    try:
        while cap.isOpened():
            ret, img = cap.read()
            cv2.imshow("My webcam", img)

            if not ret:
                break

            if(datetime.now() > finish_time):
                file = f"./images/{image_name}.png"
                cv2.imwrite(file, img)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit")
                break
            # elif key == ord(" "):
            #     print(f"Image saved at timestamp: {ts}")
            #     file = f"./images/{ts}.png"
            #     cv2.imwrite(file, img)
            #     break

    except KeyboardInterrupt:
        print("Keyboard Interrupted!")

    cap.release()
    cv2.destroyAllWindows()
