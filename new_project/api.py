from flask import Flask
# import model

app = Flask(__name__)

@app.route("/")
def hello_world(data):
    # get data from request
    cleanData = data

    # pass data to model

    # get model prediction

    # pass prediction back in response
    return "<p>Hello, World!</p>"