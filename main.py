from flask import Flask, render_template, request
import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

#global graph
#tf.get_default_graph()
model = keras.models.load_model('model_mnist.h5')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray.shape[1] > 28:
            gray28 = cv2.resize(gray, (28, 28), cv2.INTER_AREA)
        else:
            gray28 = cv2.resize(gray, (28, 28), cv2.INTER_CUBIC)



        st_mean, st_std = 33.318421449829934, 78.56748998339798
        gray28 = 255 - gray28 # инвертируем

        x = (gray28 - st_mean) / st_std
        x = x.reshape(1, 28, 28, 1)

        #print(x)

        y = model.predict(x)
        #print(y)
        y = np.argmax(y)

        return render_template('result.html', y=y)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()