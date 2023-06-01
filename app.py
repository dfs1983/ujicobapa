# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, os
from fungsi import make_model_mobile

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG', '.png', '.PNG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 6
Wayang6_classes = ["Wayang Beber", "Wayang Gedog", "Wayang Golek", "Wayang Krucil", "Wayang Kulit", "Wayang Suluh"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = './static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image_path = os.path.join(app.config['UPLOAD_PATH'], filename)
			test_image = Image.open(test_image_path)

			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((224, 224))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) 
			test_image_x       = np.array([image_array])

			#test_image_x = tf.image.resize(test_image_x, IMG_SIZE)
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = Wayang5_classes[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model_mobile()
	model.load_weights("model_wayang.h5")

	# Run Flask di localhost 
	run_with_ngrok(app)
	app.run()
