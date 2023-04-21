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


@app.route('/anemia')
def anemia():
    return render_template("anemia.html")


@app.route('/diabetesHelp')
def diabetesHelp():
    return render_template("diabetesHelp.html")


@app.route('/anemiaHelp')
def anemiaHelp():
    return render_template("anemiaHelp.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if any(value == '' for value in request.form.values()):
        return render_template('diabetes.html', pred='!! Please Enter All Values. !!')
    if any(float(value) >= 500 or float(value) < 0 for value in request.form.values()):
        return render_template('diabetes.html', pred='!! Please Enter Values in Appropriate Range. !!')
    if any(((value >= 'a' and value <= 'z') or (value >= 'A' and value <= 'Z')) for value in request.form.values()):
        return render_template('diabetes.html', pred='!! Please Enter Correct Values. !!')
    int_features = [float(x) for x in request.form.values()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = model2.transform(input_data_reshaped)
    prediction = model.predict(std_data)

    if prediction[0] == 1:
        return render_template('diabetes.html', pred='Person is Diabetic.')
    else:
        return render_template('diabetes.html', pred='Person is Non-Diabetic.')


@app.route('/predictAnemia', methods=['POST', 'GET'])
def predictAnemia():
    if any(value == '' for value in request.form.values()):
        return render_template('anemia.html', pred='!! Please Enter All Values. !!')
    if any(((value >= 'a' and value <= 'z') or (value >= 'A' and value <= 'Z')) for value in request.form.values()):
        return render_template('anemia.html', pred='!! Please Enter Correct Values. !!')
    int_features = [float(x) for x in request.form.values()]
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
