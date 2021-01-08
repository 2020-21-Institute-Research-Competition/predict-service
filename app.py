from flask import Flask
from flask import render_template
from flask import Response
import signal

from video_streaming import VideoStreaming


app = Flask(__name__)
streaming = VideoStreaming()


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


if __name__ == '__main__':
    signal.signal(signal.SIGINT, streaming.close)
    app.run(host='0.0.0.0', debug=True)
