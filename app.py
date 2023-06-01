from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Activation, Dropout, LeakyReLU
from PIL import Image
from fungsi import make_model

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG', '.png', '.PNG']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

model = None

NUM_CLASSES = 5
Wayang5_classes = ["bagong", "cepot", "gareng", "petruk", "semar"]

@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        if file_ext in app.config['UPLOAD_EXTENSIONS']:
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            test_image = Image.open('.' + gambar_prediksi)
            test_image_resized = test_image.resize((64, 64))
            image_array = np.array(test_image_resized)
            test_image_x = (image_array / 255)
            test_image_x = np.array([image_array])

            y_pred_test_single = model.predict(test_image_x)
            y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)

            hasil_prediksi = Wayang5_classes[y_pred_test_classes_single[0]]

            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
        else:
            gambar_prediksi = '(none)'
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })

if __name__ == '__main__':
    train_data = get_train_data()  # Call the function to obtain the training data
    model = make_model(train_data)  # Pass the train_data argument to make_model()
    model.load_weights("model_Wayang5_cnn_tf.h5")

    run_with_ngrok(app)
    app.run()

