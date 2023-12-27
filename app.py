from flask import Flask, render_template, url_for, jsonify, request
from preprocessing import cleaning, remove_stopwords, vectorize
import xgboost as xgb
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def beranda():
    return render_template('index.html', prediction_label=None, probability=None)
 
# Definisikan rute untuk melakukan prediksi
@app.route('/detect_scam', methods=['GET', 'POST'])
def detect_scam():
    if request.method == 'GET':
        return render_template('index.html', prediction_label=None, probability=None)
        
    # Load model Random Forest
    with open('xgb.pkl', 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    # Dapatkan teks input dari permintaan POST
    input_text = request.form['message']

    # Bersihkan teks
    cleaned_text = cleaning(input_text)

    # Hapus stopwords
    text_without_stopwords = remove_stopwords(cleaned_text)

    # Vectorize teks
    vectorized_text = vectorize([text_without_stopwords])

    # Lakukan prediksi menggunakan model Random Forest
    prediction = xgb_model.predict(vectorized_text)
    probability = xgb_model.predict_proba(vectorized_text)[:, 1]

    # Format output prediksi
    if prediction == 1:
        probability = round(float(probability) * 100, 2)
    else:
        probability = round((1 - float(probability)) * 100, 2)

    return render_template('index.html', prediction_label=prediction, probability=probability)

