import numpy as np
import pandas as pd
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# Feature extraction function
def extract_features(file_path):
    # Extract MFCC features from an audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Data preparation function
def load_data(dataset_path):
    features = []
    labels = []
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(folder_path, file)
                    mfcc = extract_features(file_path)
                    features.append(mfcc)
                    labels.append(folder)
                    
    return np.array(features), np.array(labels)

# Save features and labels as .npy files
def save_data(X, y, X_file_path, y_file_path):
    np.save(X_file_path, X)
    np.save(y_file_path, y)

# Load the dataset and preprocess
dataset_path = './TESS Toronto emotional speech set data'  # Update this with the correct dataset path
X, y = load_data(dataset_path)

# Save the features and labels into .npy files for later use
save_data(X, y, 'X.npy', 'y.npy')

# Encoding the labels
label_encoder = LabelEncoder()
y_encoded = to_categorical(label_encoder.fit_transform(y))

# Save the label encoder classes for later decoding
np.save('classes.npy', label_encoder.classes_)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model building
model = Sequential()

# Adding LSTM layers and Dense layers
model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_encoded.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Expand dimensions for LSTM input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('emotion_recognition_model.h5')

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Load and decode classes later if needed
def load_classes(class_file_path):
    return np.load(class_file_path)

# Example usage of loading classes
classes = load_classes('classes.npy')
print(f'Loaded classes: {classes}')
