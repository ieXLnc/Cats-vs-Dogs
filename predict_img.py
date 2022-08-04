import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image, extensions_file):
    if extensions_file == "jpeg" or extensions_file == "jpg":
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image


def make_predictions(filepath, extensions_file):
    # load and preprocess img
    image = tf.io.read_file(filepath)
    preprocess_img = preprocess_image(image, extensions_file)
    preprocess_img = tf.reshape(preprocess_img, [1, 224, 224, 3])
    # load model
    model = tf.keras.models.load_model("static/model/catvsdog_1.h5")
    # make the predictions and get the prob for each class
    predictions = model.predict(preprocess_img)
    predictions = predictions.flatten()
    cat_pred = predictions[0]
    dog_pred = predictions[1]
    # clear session
    keras.backend.clear_session()
    return get_label_(cat_pred, dog_pred)


def get_label_(cat_pred, dog_pred):
    label = ""
    if cat_pred > dog_pred:
        label = "cat"
        prob = cat_pred
    else:
        label = "dog"
        prob = dog_pred
    return label, np.round(prob * 100, decimals=2)
