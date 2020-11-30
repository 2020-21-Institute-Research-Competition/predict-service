from flask import Flask, request
import imageCapturer
import os

app = Flask(__name__)


@app.route("/capture", methods=["POST"])
def capture():
    json = request.get_json()
    imageCapturer.imageCapturer(
        image_name=json['id'], wait_time=5, show_label_time=2)
    return 'Done'


if __name__ == "__main__":
    app.run()
