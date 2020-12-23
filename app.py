from flask import Flask
from flask import render_template
from flask import Response
from video_streaming import VideoStreaming
import signal


app = Flask(__name__)
streaming = VideoStreaming()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(streaming.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, streaming.close)
    app.run(debug=True)
