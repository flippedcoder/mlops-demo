import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
import numpy as np
import ssl
import tensorflow as tf
import wget

app = Flask(__name__)

CORS(app)

ssl._create_default_https_context = ssl._create_unverified_context

@app.route("/classify", methods=["POST"])
def classify():
    if request.is_json:
        model = VGG16(weights="imagenet")

        # download the image file for the classifier to work
        # image_filename = "api/helianthus-yellow-flower-pixabay_11863.webp"
        print(request.json)
        image_filename = wget.download(request.json["image_url"])

        # load the image in Python Imaging Library (PIL)
        img = tf.keras.utils.load_img(image_filename, color_mode="rgb", target_size=(224, 224))

        # convert the PIL image to 3D numpy array
        x = tf.keras.utils.img_to_array(img)

        # add dimension
        x = np.expand_dims(x, axis=0)

        # pre-process image
        x = preprocess_input(x)

        # get the features
        features = model.predict(x)

        # get the predicted classes
        predictions = decode_predictions(features)[0]
        predictions.sort(key=lambda x: x[2])

        predicted_classification = predictions[-1][1]
        predicted_probability = str(predictions[-1][2])

        return jsonify(predicted_class=predicted_classification, probability=predicted_probability)

if __name__ == "__main__":
    app.run()