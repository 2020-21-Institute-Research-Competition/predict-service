from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import uuid
import json
import requests

from capture import Capture


app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        image = request.files['file']
        ID = str(uuid.uuid1().hex)
        filename = ID + secure_filename(image.filename)
        path = f'images/{filename}'
        image.save(path)

        try:
            result = json.dumps(Capture(ID, filename, path).run())
            headers = {'Content-Type': 'application/json'}
            res = requests.post(
                'http://192.168.43.142:8080/api/v1/CZon9KvrGpNouoX0rwXi/telemetry', headers=headers, data=result)
            return 'Predict successfully'
        except Exception as ex:
            print(ex)
            return 'Predict unsuccessfully'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
