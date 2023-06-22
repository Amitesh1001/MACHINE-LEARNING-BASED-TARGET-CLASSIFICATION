# MACHINE-LEARNING-BASED-TARGET-CLASSIFICATION
Classification of ground vehicles by extracting the features from their acoustic signal and finding best algorithm for the model based on accuracy using half of the signal.
## Abstract
Classification is the ability of a system to identify a sound into its specific category with an acceptable classification accuracy score. This project is able to convert the given audio to spectrograms and use these images as data to train the model to perform classification on a new audio signal. Deep learning with deep neural networks is implemented in this project which makes the system more accurate and reliable.
The main objective of this project is to devise a Deep Neural Network (DNN) for      a vehicle classification system that can categorize a given audio sample into three dif- ferent categories. A Convolutional Neural Network (CNN) and Recurrent Neural Net- work(RNN) are the type of Deep Neural Networks employed in this project, which is trained using the generated mel-spectrogram images.  With the help of Google Collab  the network is designed and imported into python. The system is designed to perform various tasks as this language provides many predefined functions such as Librosa and Sklearn giving a strong platform for implementation and provides a continuous interface between the user and the computer system.
The neural network is devised by choosing the best possible learning rate and number of iterations to train using the dataset. Softmax function is used to decide the most appropriate learning rate which is obtained by comparing the graph of the basic soft-  max function with the obtained graph. As a result, Convolutional Neural Networks and Recurrent Neural Networks are trained with an accuracy ranging between 85-95 in the process to demonstrate a comparative analysis. A vehicle classification using a deep neu- ral network is designed to perform different tasks which have environmental and security applications.
## Objectives
* To classify audio signals into Footsteps, Type-A (Amphibious Assault Vehicle) and Type-B(Dragon Wagon).
* To extract MEL spectrogram and MEL Frequency Cepstral Coefficients as features for CNN and RNN respectively.
* CNN and RNN models are trained and	 tested with full signal and half signal duration signals - to check which model is best for specific durations and denominations.
* Test and compare Convolution Neural Networks and Recurrent Neural Networks(CNN and RNN) - to evaluate which performs the best on the dataset.
## Models used in design
There are a number of features that distinguish supervised and unsupervised models, but the most integral point of difference is that how these models are trained. While supervised models are trained through examples of a particular set of data, unsupervised models are only given input data and do not have a set outcome they can learn from. While supervised models have tasks such as regression and classification and will produce a formula, unsupervised models have clustering and association rule learning. Since we will be following a supervised learning model it narrows us down to Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs).
RNN model is used to deal with unstructured data, like sequence prediction problems which include One-to-Many, Many-to-One, and Many-to-Many. CNN is mainly used for image classification purposes and help in highlighting only the important aspects of an image for classification, thereby significantly reducing the amount of data to process. Hence for our project a CNN model was selected for image classification purposes to achieve high accuracy and performance
## Design Methodology
This section describes the 2 methods which are being employed for the classification of vehicle and footstep audio signals
### Deep Neural Networks
Using Deep learning, the vehicle and footstep sound classification is done without manual feature extraction. CNN is used for the image classification.The input audio signal is represented as a spectrogram. This spectrogram is treated as an input image.The CNN uses this sprectrogram to predict the category of audio. RNN is used for classification with audio as input. Below are the detailed steps followed for the process.

**Data Generation**
One of the most critical steps is choosing the right data to train our model since the performance and the classification accuracy of the model would largely revolve around the provided data. Therefore, the data would be preprocessed to check for loss of signal data, correctness and calculating precise amount of data to feed the model avoiding overfitting. The initial raw data are sound excerpts in digital files in .mp3 format spread over three different categories; metal, grass, and concrete surfaces. These audio files in .mp3 are collectively stored in a folder, called a dataset, will be used for further processing. The dataset is divided into training and validations for classification purposes. The noisy dataset is also generated from the original audio signals by adding AWGN noise with 0.05.  <br />

**Denoising**
The input audio signal if it noisy, then the denoising is done using Discrete Wavelet Transform method.In this denosing method, the different parameters set are -
*	Mode of denoising - Soft
*	Wavelet - sym8
*	Wavelet levels - 3
*	Method for wavelet coefficient threshold selection - Visushrink  <br />
  
 **Spectrogram Generation**
Spectrograms are used for the 2D representation of the signal. Spectrograms have time on the x-axis and frequency on the y-axis.To quantify the magnitude of frequency  in a time slot, a colormap is used.In this method of classification, each audio signal is converted to mel-spectrogram. The parameters used to generate spectrograms using stft are given below:
*	Sampling rate - 22050
*	Frame/Window size(nf ft) − 2048Timeadvancebetweenframes(hoplength) = 512
*	Window function - Hanning window
*	Frequency scale - MEL
*	Number of MEL bins - 96
*	Highest frequency(fmax) = 11025(sr/2) <br />

 **MFCC Generation**
One popular audio feature extraction method is the Mel-frequency cepstral coefficients (MFCC) which have  39 features.  The feature count is small enough to force us to learn the information of the audio.  12 parameters are related to the amplitude of frequencies. It provides us enough frequency channels to analyze the audio. .mfcc is used to calculate mfccs of a signal.  By printing the shape of mfccs you  get how many mfccs are calculated on how many frames. The first value represents the number of  mfccs  calculated  and  another value represents a number of frames available.   <br />

## CNN architecture
The CNN model is implemented using TensorFlow 2.0 and Keras as API. TensorFlow
2.0 is an end-to-end, open-source machine learning platform. Its enables us to efficiently executing low-level tensor operations on CPU, GPU, or TPU. Keras is the high-level  API of TensorFlow 2.0: an approachable, highly productive interface for solving machine learning problems, with a focus on modern deep learning. It provides essential abstrac- tions and building blocks for developing and shipping machine learning solutions with high iteration velocity.
First,a sequential model is used, starting with a simple model architecture, consisting of three Conv2D convolution layers,  with our final output layer  being a dense layer.  The output layer uses Softmax function as an activation layer, having 3 nodes which matches the number of possible classifications. The images are normalized by dividing each individual pixel with a constant number to bring it in the format of ones and zeros. The ImageDataGenerator functions, for training and validations, provided by the Keras API performs the automatic labelling of the dataset referring the structure of the fed dataset.
The specifications of the train ImageDataGenerator are calculated according to their respective formulas, batch size as 10, class mode set as categorical, target size as (40,61), and the path of the train spectrogram images. While the specifications of the validations ImageDataGenerator are calculated as, batch size as 10, class mode set as categorical, target size as (40,61), and the path of the validations spectrogram images. The model used categorical cross entropy as the loss function and optimizer as the RMSprop function for which the learning rate is set as 0.001. The model is run for 50 epochs(50 steps per each epoch), on each epoch the training and validation accuracies are noted and observed for convergency and higher convergence accuracy on each step.

## LSTM RNN architecture
One of the most popular recurrent neural nets is Long short-term Memory or LSTM neural network, which played an important role in solving the vanishing gradient problem of recurrent neural nets. LSTM is also the building block of many applications in areas of machine translation, speech recognition and other tasks, where LSTM networks achieved considerable success. The key concept of the LSTM is cell state or the “memory state”  of the network, which captures information from previous steps. Information is added to the cell state with several different gates: forget gate, input gate and output gate. Gates can be thought of as control units that control which data is added to the cell state.    Next, previous hidden state is passed and the input to the sigmoid of the input gate and also pass hidden state and current input to the tanh function, followed by multiplying both sigmoid and tanh output.All these values are then used to update the cell state, by first multiplying cell state by vector from the forget gate. This is then pointwise added to the vector from the input gate, to obtain the new, updated value  of the cell state.LSTM  is then concluded with the final, output gate. Its output is computed by first passing previous hidden state and the input to the sigmoid function and then multiplying this  with the updated state that was passed to the tanh function.  The output is the new  hidden state which is passed to the next time step along with the new cell state. Neural net consists of an embedding layer, LSTM layer with 64 memory units and a Dense output layer with one neuron and a sigmoid activation function,trained for 100 epochs and evaluate it on the test set. 
  



