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