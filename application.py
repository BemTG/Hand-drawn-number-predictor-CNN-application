#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import logging, sys
logging.basicConfig(level=logging.DEBUG)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os 
import numpy as np
from array import array
from base64 import decodestring
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])
    
    return model

# global variables
application=app = Flask(__name__)

# full path to check point
tensorflow_ckpt_file = 'cp.ckpt'

def GetDigitPrediction(img):
    # restore the saved session
    model = create_model()
    model.load_weights(tensorflow_ckpt_file)
    prediction = model.predict_classes(img, batch_size=1)
    return(prediction[0])


@app.route("/", methods=['POST', 'GET'])
def DrawAndPredict():
    drawing_data = ''
    prediction = '?'

    # set up page with blanks
    return render_template('predict_number.html',
        drawing_data = drawing_data,
        prediction = prediction)

@app.route('/background_process', methods=['POST', 'GET'])
def background_process():
    prediction = -1
    try:
        drawing_data_original = request.form['drawing_data'] 
        user_drawn_image = drawing_data_original.split(',')[1]
        if len(user_drawn_image) > 0:
            buf = io.BytesIO(base64.b64decode(user_drawn_image))
            img = Image.open(buf)
            img = img.resize([28,28])


            # these are transparent images so apply a white background
            corrected_img = Image.new("RGBA", (28, 28), "white")
            corrected_img.paste(img, (0,0), img)
            corrected_img = np.asarray(corrected_img)

            # remove color dimensions
            corrected_img = corrected_img[:, :, 0]
            img = np.invert(corrected_img)
            img=img.astype('float32')
            #centre the image and normalize the data for the data to learn better
            img=img/255


            logging.info(img) 
            # reshape for model to (1, 28, 28, 1)
            img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
            prediction = int(GetDigitPrediction(img))
            logging.info(prediction)
    except:  
        # something didn't go right
        e = sys.exc_info()[0]
        prediction = e 
        logging.info(e)

    return jsonify({'prediction':prediction})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=80,debug=True)