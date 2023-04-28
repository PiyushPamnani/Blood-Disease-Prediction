from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

anemia_model = pickle.load(open('anemia_model.pkl', 'rb'))
anemia_model2 = pickle.load(open('anemia_model2.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("diabetes.html")


@app.route('/diabetesParameters')
def diabetesParameters():
    return render_template("diabetesParameters.html")


@app.route('/diabetesHelp')
def diabetesHelp():
    return render_template("diabetesHelp.html")


diabetes_param_ranges = {
    'Pregnancies': [0, 15],
    'Glucose': [60, 200],
    'Blood Pressure': [0, 150],
    'Skin Thickness': [0, 100],
    'Insulin': [0, 900],
    'BMI': [0, 70],
    'Diabetes Pedigree Function': [0, 3],
    'Age': [18, 90]
}


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    # Check the validity of each input value
    for param, range_ in diabetes_param_ranges.items():
        value = request.form.get(param)
        try:
            value = float(value)
            if value < range_[0] or value > range_[1]:
                return render_template('diabetes.html', pred=f'{param.title()} value should be between {range_[0]} and {range_[1]}')
        except ValueError:
            return render_template('diabetes.html', pred=f'Invalid {param.title()} value')
    int_features = [float(request.form.get(param))
                    for param in diabetes_param_ranges.keys()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = model2.transform(input_data_reshaped)
    prediction = model.predict(std_data)

    if prediction[0] == 1:
        return render_template('diabetes.html', pred='Person is Diabetic.')
    else:
        return render_template('diabetes.html', pred='Person is Non-Diabetic.')


@app.route('/anemia')
def anemia():
    return render_template("anemia.html")


@app.route('/anemiaParameters')
def anemiaParameters():
    return render_template("anemiaParameters.html")


@app.route('/anemiaHelp')
def anemiaHelp():
    return render_template("anemiaHelp.html")


anemia_param_ranges = {
    'Gender': [0, 1],
    'Hemoglobin': [5, 20],
    'MCH': [10, 40],
    'MCHC': [15, 50],
    'MCV': [60, 110],
}


@app.route('/predictAnemia', methods=['POST', 'GET'])
def predictAnemia():
    # Check the validity of each input value
    for param, range_ in anemia_param_ranges.items():
        value = request.form.get(param)
        try:
            value = float(value)
            if value < range_[0] or value > range_[1]:
                return render_template('anemia.html', pred=f'{param.title()} value should be between {range_[0]} and {range_[1]}')
        except ValueError:
            return render_template('anemia.html', pred=f'Invalid {param.title()} value')
    int_features = [float(request.form.get(param))
                    for param in anemia_param_ranges.keys()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = anemia_model2.transform(input_data_reshaped)
    prediction = anemia_model.predict(std_data)

    if prediction[0] == 1:
        return render_template('anemia.html', pred='Person is Anemic.')
    else:
        return render_template('anemia.html', pred='Person is Not Anemic.')


if __name__ == '__main__':
    app.run(debug=True)
