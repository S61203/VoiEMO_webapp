import os
import librosa
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('emotion_recognition_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Function to extract features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        if 'file' not in request.files:
            return "Please upload a file"
    
        file = request.files['file']
    
        if file.filename == '':
            return "No selected file"
    
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
    
            # Extract features and make prediction
            features = extract_features(file_path)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)

            prediction = model.predict(features)
            predicted_emotion = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

            return jsonify({'emotion': predicted_emotion[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
