from flask import Flask, request, render_template
import imageCapturer
import os

app = Flask(__name__)


@app.route("/capture", methods=["POST"])
def capture():
    json = request.get_json()
    imageCapturer.imageCapturer(json['id'], 5)
    return 'Done'


if __name__ == "__main__":
    app.run()
