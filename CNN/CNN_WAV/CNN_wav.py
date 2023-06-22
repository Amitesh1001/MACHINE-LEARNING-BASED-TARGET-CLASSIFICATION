from google.colab import drive
drive.mount('/content/drive')
import json
import os
import math
import librosa
from keras.models import model_from_json
#from GenreFeatureData import (    GenreFeatureData,)  # local python class with Audio feature extraction and genre list

# set logging level
#logging.getLogger("tensorflow").setLevel(logging.ERROR)

'''
def load_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model

'''
SAMPLE_RATE = 22050
TRACK_DURATION = 20 # measured in seconds
num_segments=3
hop_length=512
num_mfcc=13
n_fft=2048
#json_path="data__.json"
data1 = {
         "mfcc": []
    }
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
file_path="dataset"
def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    signal, sample_rate = librosa.load(file, sr=SAMPLE_RATE)

                # process all segments of audio file
    for d in range(num_segments):

                    # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

                    # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data1["mfcc"].append(mfcc.tolist())
            #data1["labels"].append()
            #print("{}, segment:{}".format(file, d+1))
        # save MFCCs to json file
    #with open(json_path, "w") as fp:
        #json.dump(data1, fp, indent=4)
        return data1["mfcc"]


def print_prediction(file_name):
    prediction_feature = extract_audio_features(file_name)
    # y=data1["labels"]
    # predicted_vector = model.predict_classes(prediction_feature)

    # predicted_class = le.inverse_transform(predicted_vector)
    # print("The predicted class is:", predicted_class[0], '\n')
    pf = np.array(prediction_feature)
    # print(pf.shape)
    pf = pf[..., np.newaxis]

    # predicted_proba = predicted_proba_vector[0]

    # for i in range(len(predicted_proba)):
    #   category = le.inverse_transform(np.array([i]))
    #   print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
    # prediction
    # prediction = model.predict(prediction_feature)
    # get index with max value
    # predicted_index = np.argmax(prediction, axis=1)
    # print(predicted_index[-1])
    # print(predicted_proba_vector[0])
    # predicted_proba = predicted_proba_vector[0]

    prediction = model.predict(pf)
    predicted_index = np.argmax(prediction, axis=1)
    if (predicted_index[-1] == 1):
        print("aav")
    elif (predicted_index[-1] == 3):
        print("dw")
    elif (predicted_index[-1] == 5):
        print("footstep")
