from flask import Flask, render_template, request, redirect, url_for
from keras.preprocessing.image import load_img
import joblib
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16


def getResult(image):
    SIZE = 224

    # preprocessing
    image = np.array(image)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # feature extraction
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    for layer in VGG_model.layers:
        layer.trainable = False
    feature_extractor = VGG_model.predict(image)
    Image_features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    X_for_SVM = Image_features

    # predict the image using trained SVM
    result = classifier.predict(X_for_SVM)

    return result

def get_className(classNo):
	if classNo==0:
		return "fake"
	elif classNo==1:
		return "real"

classifier = joblib.load('Deepfake detector.pkl')
app = Flask(__name__, template_folder='template', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        image = load_img(uploaded_file.filename, target_size=(224, 224))
        Class_no = getResult(image)
        Classification = get_className(Class_no)
    return render_template('index.html', prediction= Classification)

if __name__ == '__main__':
   app.run(debug = True)