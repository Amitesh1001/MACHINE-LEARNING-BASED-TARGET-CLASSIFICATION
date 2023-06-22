import numpy as np
from skimage.restoration import denoise_wavelet
from google.colab import drive
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
def plotsignal(parent_dir,sub_dir,file_ext="*.wav"):
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        data, sr = librosa.load(fn, sr = 22050,mono = True)
        dsound_clip=denoise_wavelet(data,method='VisuShrink',mode='soft',
                                    wavelet_levels=3,wavelet='sym8',
                                    rescale_sigma='True')
        plt.title('Denoised Audio signal wrt time')
        librosa.display.waveplot(dsound_clip,sr=sr)
        ##plt.figure();

drive.mount('/content/gdrive/')
parent_dir = 'gdrive/My Drive/Universal_Dataset'
sub_dirs = np.array(['Footsteps','TYPE_A','TYPE_B'])

for sub_dir in sub_dirs:
    plotsignal(parent_dir,sub_dir)
from google.colab import drive
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_wavelet
from google.colab import drive
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
def plotsignal(parent_dir,sub_dir,file_ext="*.wav"):
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        data, sr = librosa.load(fn, sr = 22050,mono = True)
        dsound_clip=denoise_wavelet(data,method='VisuShrink',mode='soft',
                                    wavelet_levels=3,wavelet='sym8',
                                    rescale_sigma='True')
        plt.title('Denoised Audio signal wrt time')
        librosa.display.waveplot(dsound_clip,sr=sr)
        ##plt.figure();

drive.mount('/content/gdrive/')
parent_dir = 'gdrive/My Drive/Universal_Dataset'
sub_dirs = np.array(['Footsteps','TYPE_A','TYPE_B'])

for sub_dir in sub_dirs:
    plotsignal(parent_dir,sub_dir)

def scale_minmax(X, min=0.0, max=1.0):
    x_std = (X - X.min()) / (X.max() - X.min())
    x_scaled = x_std * (max - min) + min
    return x_scaled
def getLabel(fn):
    if fn.split('/')[3] == 'TYPE_A':
        label = 'A'
    elif fn.split('/')[3] == 'TYPE_B':
        label = 'B'
    else:
        label = 'H'
    return label
def _windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start+=(window_size//2)
drive.mount('/content/gdrive/')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(41, 60, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
def audiostozip(parent_d,sub_d):
    parent_dir = parent_d
    sub_dirs = sub_d
    for sub_dir in sub_dirs:
      saveSpec(parent_dir,sub_dir)

    shutil.make_archive('gdrive/My Drive/Universal_Dataset/Universal_Spectrograms', 'zip', 'gdrive/My Drive/Universal_Dataset/Universal_Spectrograms')

parent_dir = 'gdrive/My Drive/Universal_Dataset'
sub_dirs = np.array(['Footsteps','TYPE_A','TYPE_B'])
audiostozip(parent_dir,sub_dirs)

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255)
Validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Universal_Dataset /Universal_Dataset/Universal_Spectrograms_CNN/Train/',
    target_size=(40, 61),
    batch_size=10,
    class_mode='categorical')

Validation_generator = Validation_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Universal_Dataset /Universal_Dataset/Universal_Spectrograms_CNN/Validations/',
    target_size=(40, 61),
    batch_size=10,
    class_mode='categorical')

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=50,
                    verbose=1,
                    validation_data=Validation_generator,
                    validation_steps=50
                    )
print(model.metrics_names)
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

classe = train_generator.class_indices
print(classe)
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path, target_size=(40, 61))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    # print(classes)

    if classes[0][0] == 1.0:
        print("Human Footstep")
    elif classes[0][1] == 1.0:
        print("TYPE_A")
    elif classes[0][2] == 1.0:
        print("TYPE_B")
    else:
        print("Audio is not recognised")
# Importing libraries
%matplotlib inline
import pandas as pd
import seaborn as sns
import numpy
import numpy as np
from matplotlib import pyplot as plt
import librosa, librosa.display
from scipy.signal import resample

# Sampling frequency
sr=22050

# Reading the data as a dataframe

df1=pd.read_csv("/content/drive/MyDrive/Full Signals/A-type/A10_21_S.csv",names=['amplitude', 'column2','column3','column4'])

# Dropping columns which contains missing values
df2 = df1.drop(['column2','column3','column4'], axis=1)

# Plotting the original signal
plt.figure(figsize= (12,8))
plt.subplot(3,1,1)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Original signal')
plt.plot(df2)

# Converting dataframe to numpy array
am1=df2.to_numpy()
am=am1.ravel()
# Display Spectrogram

X = librosa.stft(am)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 3))
#tr=librosa.display.specshow(Xdb, sr=22050, x_axis='time', y_axis='hz')
tr=librosa.display.specshow(Xdb, sr=22050)
#If to pring log of frequencies
#librosa.display.specshow(Xdb, sr=88200, x_axis='time', y_axis='log')
plt.colorbar();
#plt.title('Spectrogram')
plt.savefig('changed_from_csv_spect.png', dpi=100)