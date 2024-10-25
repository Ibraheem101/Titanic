import numpy as np
import pickle
from flask_cors import CORS
from flask import Flask, request, jsonify

model = pickle.load(open('titanic_model.pkl', 'rb'))
columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'female', 'male', 'Cherbourg', 'Queenstown', 'Southampton']


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = []
    for column in columns:
        globals()[column] = data[column]
        features.append(globals()[column])
    features = np.array([features]).astype(float)

    prediction = model.predict(features)[0]
    result = {
        'Survived': bool(prediction),  # Convert to boolean to make it clearer
        'Message': 'Passenger survived' if prediction == 1 else 'Passenger did not survive'
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
