import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

from flask import url_for
from flask_api import FlaskAPI, status, exceptions
from models import sessions
from datetime import datetime
from models import notes
from bson import ObjectId

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_features = [float(x) for x in request.form.values()]
    post_features = [np.array(form_features)]
    prediction = model.predict(post_features)
    print('{}'.format(prediction[0]))

    return render_template('index.html', prediction_text='Heart disease target {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
