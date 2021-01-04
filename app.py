from flask import Flask
from flask import render_template
from flask import Response
from flask import request
from flask import jsonify
import signal
import csv

from video_streaming import VideoStreaming
from capture import Capture


app = Flask(__name__)
streaming = VideoStreaming()
Capture().start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(streaming.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop')
def stop():
    return streaming.close(None, None)


@app.route('/lastest-predictions')
def latest():
    nums = int(request.args.get('nums', 5))
    lastest_pred = {'predictions': []}

    with open('predicted_results.csv', 'r') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        count = 0

        if nums > len(csv_reader):
            nums = len(csv_reader)

        for line in csv_reader:
            if count > nums:
                break

            lastest_pred['predictions'].append(
                {'id': line[0], 'status': line[1], 'image_url': line[2]})
            count += 1

    return jsonify(lastest_pred)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, streaming.close)
    app.run(host='0.0.0.0', debug=True)
