import cv2
import numpy as np
import base64
from flask import Flask,render_template,url_for,request
import tensorflow.keras
import re

app = Flask(__name__, template_folder='templates')

model = tensorflow.keras.models.load_model('model_hand.h5')

char_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

@app.route("/")
def home():
    return render_template('draw.html')

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        #Accessing the image
        draw = request.form['url']
        draw = re.sub('^data:image/.+;base64,', '', draw)
        draw_decoded = base64.b64decode(draw)

        print(draw)

        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        img_copy = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (400, 440))

        img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
        _, img_thresh = cv2.threshold(img_copy, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        # Launch prediction
        my_prediction = model.predict(img_final)
        # Getting the index of the maximum prediction
        index = np.argmax(my_prediction[0])
        # Associating the index and its value within the dictionnary
        final_pred = char_dict[index]

    return render_template('results.html', prediction=final_pred)


if __name__ == '__main__':
    app.run(debug=True)