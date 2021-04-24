from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np

model = pickle.load(open("D:\Project\my project\heart_disease.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")


@app.route('/predict', methods = ['POST'])
def predict():


    final_feat = list(request.form.values())
    final_feat = [float(numeric_string) for numeric_string in final_feat]
    final_feat = np.array(final_feat).reshape(1,13)

    output = model.predict(final_feat)[0]

    return render_template("predict.html",output=output)


if __name__=='__main__':
    app.run(debug=True)