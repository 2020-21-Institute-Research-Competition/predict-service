from time import time, sleep
from datetime import datetime, timedelta
import cv2
from pathlib import Path

Path("./images").mkdir(parents=True, exist_ok=True)

# * If we have a tuple of starting coordinate(5, 5) and ending coordinate(220, 220) then
# *    (5,5)                (5, 220)
# *      ***********************
# *      *                     *
# *      *                     *
# *      *                     *
# *      *                     *
# *      *                     *
# *      ***********************
# *  (220, 5)              (220, 220)


def imageCapturer(image_name, wait_time, show_label_time):
    cap = cv2.VideoCapture(0)
    ts = int(time())
    label_time = datetime.now() + timedelta(seconds=show_label_time)
    finish_time = datetime.now() + timedelta(seconds=wait_time)

    try:
        while cap.isOpened():
            ret, img = cap.read()

            height, width, channel = img.shape
            center_width = width // 2
            center_height = height // 2
            starting_point = (center_width - 100, center_height - 150)
            ending_point = (center_width + 100, center_height + 150)
            text_org = (center_width - 80, center_height - 170)

            if(datetime.now() > label_time):
                cv2.rectangle(img, starting_point,
                              ending_point, (0, 255, 0), 2)
                cv2.putText(img, 'Black bean plant', text_org,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

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
