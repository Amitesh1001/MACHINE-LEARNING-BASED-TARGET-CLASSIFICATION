'''import pandas as pd
import soundfile as sf
import os
import wave
# assume we have columns 'time' and 'value'

 # df = pd.read_csv('/content/drive/MyDrive/Full Signals/Full Signals/B-Type/B5_21_S.csv')

local_download_path = os.path.expanduser('/content/drive/MyDrive/Full Signals/Full Signals/A-type')
#try: os.makedirs(local_download_path)
#except: pass

data = []
#for file in os.listdir(local_download_path):
 # data_read.append(pd.read_csv(file))
data = os.listdir(local_download_path)
data_read = []
data_write = []
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
for i in data:
  #if filename.endswith("csv"):
    data_read[i] = pd.read_csv(data[i])
   # data_write[i] = (convert(data(filename)))
    wave.open(convert(data(filename)), mode=None)
# compute sample rate, assuming times are in seconds
times = df['time'].values
n_measurements = len(times)
timespan_seconds = times[-1] - times[0]
sample_rate_hz = int(n_measurements / timespan_seconds)'''

# #write data
#data = df[:].values
##sf.write('new_data.wav', data, sample_rate_hz)
#sf.write('new_data.wav', data, 22050)
#print(data_write)
'''for file in data_read:
  data = data_read(file)
  data_write.append(sf.write('new_data.wav', data, 22050))'''



def convert(csv_data):
    data_final = csv_data[:].values
    sf.write('new_data.wav', data_final, 22050)



import pandas as pd
import soundfile as sf
import os
import wave
# assume we have columns 'time' and 'value'

 # df = pd.read_csv('/content/drive/MyDrive/Full Signals/Full Signals/B-Type/B5_21_S.csv')

dataset_path = os.path.expanduser('/content/drive/MyDrive/Full Signals/Full Signals')
j = 0
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    for f in filenames:
                convert(pd.read_csv(f))


filename = 'new_data.wav'
print_prediction(filename)
#prediction_class = le.inverse_transform(filename)
#prediction_class