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

# === REAL FOURIER DICTIONARY CREATION ===
def create_fourier_dictionary_real(N, M=None):
    """
    Creates a real-valued Fourier dictionary for signal decomposition.

    Args:
        N (int): Signal/frame length.
        M (int): Number of dictionary atoms (defaults to 2*N).

    Returns:
        np.ndarray: Normalized real-valued Fourier dictionary of shape (N, M).
    """
    if M is None:
        M = 2 * N
    t = np.arange(N).reshape(-1, 1)
    freqs = np.linspace(0, np.pi, M // 2, endpoint=False)

    # Construct cosine and sine bases
    Phi_cos = np.cos(t * freqs)
    Phi_sin = np.sin(t * freqs)
    Phi = np.concatenate((Phi_cos, Phi_sin), axis=1)

    # Normalize dictionary atoms
    norms = np.linalg.norm(Phi, axis=0, keepdims=True)
    norms[norms == 0] = 1
    Phi /= norms
    return Phi

# === MATCHING PURSUIT IMPLEMENTATION ===
def matching_pursuit(Phi, signal, max_iters=20):
    """
    Performs Matching Pursuit using a real Fourier dictionary.

    Args:
        Phi (np.ndarray): Dictionary matrix of shape (N, M).
        signal (np.ndarray): Input signal of shape (N,).
        max_iters (int): Number of iterations.

    Returns:
        np.ndarray: Reconstructed signal using sparse approximation.
    """
    N, M = Phi.shape
    residual = signal.astype(np.float64)
    coeffs = np.zeros(M, dtype=np.float64)

    for _ in range(max_iters):
        inner_products = np.dot(Phi.T, residual)
        atom_index = np.argmax(np.abs(inner_products))
        coeff = inner_products[atom_index]

        coeffs[atom_index] += coeff
        residual -= coeff * Phi[:, atom_index]

        if np.linalg.norm(residual) < 1e-6:
            break

    return np.dot(Phi, coeffs)

# === CREATE DICTIONARY ===
N = frame_length
M = 2 * N
Phi = create_fourier_dictionary_real(N, M)

# === MATCHING PURSUIT ON ALL FRAMES ===
reconstructed_frames = np.zeros_like(frames)
for i in range(frames.shape[0]):
    reconstructed_frames[i] = matching_pursuit(Phi, frames[i], max_iters=20)

# === OVERLAP-ADD FUNCTION ===
def overlap_add(frames, frame_length, hop_length):
    """
    Reconstructs the time-domain signal from overlapping frames using the overlap-add method.

    Args:
        frames (np.ndarray): Array of overlapping frames (shape: [num_frames, frame_length]).
        frame_length (int): Length of each frame.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Reconstructed 1D signal.
    """
    num_frames = frames.shape[0]
    signal_length = hop_length * (num_frames - 1) + frame_length
    reconstructed = np.zeros(signal_length)
    window = np.hanning(frame_length)

    for i in range(num_frames):
        start = i * hop_length
        reconstructed[start:start + frame_length] += frames[i] * window
    return reconstructed

# === RECONSTRUCT FULL AUDIO ===
y_reconstructed = overlap_add(reconstructed_frames, frame_length, hop_length)

# === CALCULATE RECONSTRUCTION ERROR ===
min_len = min(len(y), len(y_reconstructed))
error = np.linalg.norm(y[:min_len] - y_reconstructed[:min_len]) / np.linalg.norm(y[:min_len])
print("Total relative reconstruction error:", error)

# === PLOT ORIGINAL VS RECONSTRUCTED SIGNAL ===
plt.figure(figsize=(10, 4))
plt.plot(y[:5000], label="Original Signal")
plt.plot(y_reconstructed[:5000], label="Reconstructed Signal", linestyle='--')
plt.legend()
plt.title("Full Audio Matching Pursuit Reconstruction (Zoomed In)")
plt.tight_layout()
plt.show()

