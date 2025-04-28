import librosa
import numpy as np
import cr.sparse.dict as dl
import jax.numpy as jnp
from cr.sparse.pursuit import omp, mp, sp
from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#filename = librosa.ex('trumpet')
filename = "./dataset/1727.wav"
# Load audio
y, sr = librosa.load(filename, sr=None)

# We create 2 versions of PCA, one for the MFCCS and one for the basic frames of the audio

# Frame the signal
frame_length = 1024
hop_length = 512
frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T)
frame_scaled = scaler.fit_transform(frames.T)

print(mfccs_scaled.max())
print(frame_scaled.max())

pca = PCA(n_components=2)
principal_components_mfccs = pca.fit_transform(mfccs_scaled)
principal_components_frames = pca.fit_transform(frame_scaled)


print(principal_components_mfccs.max())
print(principal_components_frames.max())

#print(mfccs)

plt.subplot(1, 2, 1)
plt.scatter(principal_components_mfccs[:, 0],principal_components_mfccs[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA MFCCS')

plt.subplot(1, 2, 2)
plt.scatter(principal_components_frames[:, 0],principal_components_frames[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA FRAMES')


plt.figure()
#librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar(librosa.display.specshow(mfccs, x_axis='time'))
plt.title('MFCC')
plt.tight_layout()
plt.show()