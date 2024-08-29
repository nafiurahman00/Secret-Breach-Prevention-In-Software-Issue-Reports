from flask import Flask, request
from flask import send_from_directory
import pandas as pd
from secret_finding_for_tool import prediction
from flask_cors import CORS
UPLOAD_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, resources={r"/checkdescription": {"origins": "*"}})  # Replace "*" with the specific origin if needed

# receive json and predict if it contains secrets return true or false
@app.route('/checkdescription', methods=['POST'])
def predict_endpoint():
    payload = request.json
    print(payload)
    description = payload['description']
    dict ={}
    dict[0] = {'body': description}
    data = pd.DataFrame.from_dict(dict, "index")
    data.to_csv('issue.csv', index=False)     
    df = pd.read_csv('issue.csv')
    content = df['body'][0]
    # print(secret_finding_for_tool.prediction(content))
    result,result_strings = prediction(content)
    print(result)
    print(result_strings)
    return {'prediction': result,
            'candidates':result_strings}

@app.route('/.well-known/pki-validation/<path:path>')
def verify(path):
    print(path)
    return send_from_directory('static',path)

if __name__ == 'main':
    app.run(host="0.0.0.0",port=5000,ssl_context=('certificate.crt', 'private.key'))