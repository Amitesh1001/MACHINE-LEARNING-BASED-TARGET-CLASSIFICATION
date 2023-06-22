import json
import os
import math
import librosa

dataset_path = "/content/gdrive/MyDrive/Deep Learning/RNN_Wav/Training dataset"
json_path = "/content/gdrive/MyDrive/Deep Learning/RNN_Wav/data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 0.1  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=2):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        #########################################################################
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                signal1 = signal[:len(signal) // 2]

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = (start + samples_per_segment)

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(dataset_path, json_path, num_segments=2)
#y_predict=model.predict_classes(X_test)
#y_predict
predict_x=model.predict(X_test)
y_predict=np.argmax(predict_x,axis=1)
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
le=LabelEncoder()
#label_enc = LabelEncoder()
#le.fit(y)
y_predict
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "/content/gdrive/MyDrive/Deep Learning/RNN_Wav/data_10.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """
    # y_predict1=model.predict_classes(X_train)
    # y_predict1
    predict_x = model.predict(X_train)
    y_predict1 = np.argmax(predict_x, axis=1)
    y_predict1
    # build network topology
    model = keras.Sequential()

    # 1 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])  # 130, 13
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    X_test
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_predict)))



ax= plt.subplot()
cm=metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['A-Type', 'B-Type']); ax.yaxis.set_ticklabels(['A-Type', 'B-Type']);
plt.show()
cm

#df_cm = pd.DataFrame(cm, index = [i for i in ["0","1"]],
                  #columns = [i for i in ["Predict 0","Predict 1"]])
#sns.heatmap(df_cm, annot=labels, fmt='g');

#y_test
sum=0
for i in range(0,len(X_test)):
    if(y_predict[i]==y_test[i]):
        sum+=1
print(sum/len(X_test))
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
TRACK_DURATION = 0.1 # measured in seconds
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
file_path="/content/drive/MyDrive/Deep Learning/RNN_Wav/Training dataset"
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

    predicted_proba_vector = model.predict(prediction_feature)
    # predicted_proba = predicted_proba_vector[0]

    # for i in range(len(predicted_proba)):
    #   category = le.inverse_transform(np.array([i]))
    #   print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
    # prediction
    prediction = model.predict(prediction_feature)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    # print(predicted_index[-1])
    # print(predicted_proba_vector[0])
    # predicted_proba = predicted_proba_vector[0]
    if (predicted_index[-1] == 1):
        print("A-Type")
    elif (predicted_index[-1] == 0):
        print("B-Type")
    # elif(predicted_index[-1]==2):
    # print("SEISMIC_AAV")
    # elif(predicted_index[-1]==3):
    #   print("SEISMIC_DW")

filename = 'new_data.wav'
print_prediction(filename)
#prediction_class = le.inverse_transform(filename)
#prediction_class

import json
import os
import math
import librosa

DATASET_PATH = "/content/gdrive/MyDrive/Deep Learning/Signals/Original data"
JSON_PATH = "data_11.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 0.1  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=3)