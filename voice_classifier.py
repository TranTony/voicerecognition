"""
This example is to compare 2 human voice using Logistic Regression and Decision Tree. You could see the coding style was so terrible because it's trial.
"""

import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
from speechpy import processing
from speechpy import feature
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/x.wav')
file_name2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/y.wav')
fs, signal = wav.read(file_name)
fs2, signal2 = wav.read(file_name2)

#signal = signal[:,0]
#signal2 = signal2[:,0]
# Pre-emphasizing.
signal_preemphasized = processing.preemphasis(signal, cof=0.98)
signal_preemphasized2 = processing.preemphasis(signal2, cof=0.98)


# Staching frames
frames = processing.stack_frames(signal, sampling_frequency=fs,
                                          frame_length=0.020,
                                          frame_stride=0.01,
                                          filter=lambda x: np.ones((x,)),
                                          zero_padding=True)
# Staching frames
frames2 = processing.stack_frames(signal2, sampling_frequency=fs,
                                          frame_length=0.020,
                                          frame_stride=0.01,
                                          filter=lambda x: np.ones((x,)),
                                          zero_padding=True)



# Extracting power spectrum
power_spectrum = processing.power_spectrum(frames, fft_points=512)
print('power spectrum shape=', power_spectrum.shape)

############# Extract MFCC features #############
mfcc_feat= feature.mfcc(signal, sampling_frequency=fs,
                             frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0,
                             high_frequency=None)
mfcc_feat2= feature.mfcc(signal2, sampling_frequency=fs,
                             frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0,
                             high_frequency=None)


# Cepstral mean variance normalization.
mfcc_cmvn = processing.cmvn(mfcc_feat,variance_normalization=True)

mfcc_cmvn2 = processing.cmvn(mfcc_feat2,variance_normalization=True)

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(mfcc_feat)
scaled_train_features2 = scaler.fit_transform(mfcc_feat2)

# Get our explained variance ratios from PCA using all features
#pca = PCA()
#pca.fit(scaled_train_features)
n_components = 6
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)
print('pca projection', pca_projection)

pca2 = PCA(n_components, random_state=10)
pca2.fit(scaled_train_features2)
pca_projection2 = pca2.transform(scaled_train_features2)
print('pca projection2', pca_projection2)
#Showing mfcc_feat

import numpy as np

mfcc_flat = pca_projection.ravel()
mfcc_flat2 = pca_projection2.ravel()
print("flat:", mfcc_flat.shape)
print("flat2:", mfcc_flat2.shape)
dist = np.linalg.norm(np.array(mfcc_flat[10000:20000])-np.array(mfcc_flat2[10000:20000]))
dist2 = np.linalg.norm(np.array(mfcc_flat2[:10000])-np.array(mfcc_flat2[10000:20000]))
print("E dist:", dist)
print("E dist2:", dist2)
#print("flat shape:", mfcc_flat.shape)
#plt.plot(mfcc_flat, 'gx')
#plt.plot(mfcc_flat2, 'rx')
#plt.show()

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Train our logistic regression and predict labels for the test set
logreg = LogisticRegression(random_state=10)



train_features = [mfcc_flat[8000:9000], mfcc_flat2[8000:9000], mfcc_flat[7000:8000], mfcc_flat2[7000:8000]]
train_labels = ['x', 'y', 'x', 'y']

test_features = [mfcc_flat2[1000:2000], mfcc_flat[1000:2000], mfcc_flat2[2000:3000], mfcc_flat[2000:3000], mfcc_flat2[3000:4000], mfcc_flat[3000:4000], mfcc_flat2[4000:5000], mfcc_flat[4000:5000], mfcc_flat2[5000:6000], mfcc_flat[5000:6000]]
test_labels = ['y','x','y', 'x','y','x','y', 'x', 'y', 'x']

logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)
pred_labels_tree = tree.predict(test_features)
print('label_test', test_labels)
print('label_tree', pred_labels_tree)
print('label_logit', pred_labels_logit)
class_rep_tree = classification_report(test_labels, pred_labels_tree)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))


print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

# Extracting derivative features
mfcc_feature_cube = feature.extract_derivative_feature(mfcc_feat)

print('mfcc feature cube shape=', mfcc_feature_cube.shape)

############# Extract logenergy features #############
logenergy = feature.lmfe(signal, sampling_frequency=fs,
                                  frame_length=0.020, frame_stride=0.01,
                                  num_filters=40, fft_length=512,
                                  low_frequency=0, high_frequency=None)
logenergy_feature_cube = feature.extract_derivative_feature(logenergy)
print('logenergy features=', logenergy.shape)





