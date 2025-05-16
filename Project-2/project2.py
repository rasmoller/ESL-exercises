import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import SparseCoder
from sklearn.preprocessing import normalize

from fourier_dictionary import FourierDictionary
from learned_dictionary import AudioDictionary

n_atoms = 32
signal_length = 128
frame_length = 1024
frame_count = 800

FD = FourierDictionary(frame_length).create_fourier_dictionary()
LD = AudioDictionary(frame_length, frame_count).learn_dictionary()

# Normalize the dictionaries
FD = normalize(FD, axis=0)
LD = normalize(LD, axis=0)

def signal_generator(filepath="./dataset/1727.wav"):
    y, _ = librosa.load(filepath, sr=None)
    return y

def sparse_approximation(signal, dictionaries):
    results = dict()

    for idx, dictionary in enumerate(dictionaries):
        for algo in ["omp", "lasso_cd", "lars"]:
            # Create a composite key using the algorithm name and dictionary index
            composite_key = f"{algo}_dict_{idx}"
            print("Training " + composite_key)

            coder = SparseCoder(dictionary=dictionary, transform_algorithm=algo) # type: ignore
            if idx == 0:
                results[composite_key] = coder.transform(signal)
            else:
                results[composite_key] = coder.fit_transform(signal)

            print("Done training " + composite_key)
    return results

if __name__ == "__main__":
    original_signal = signal_generator()[:(frame_length//2)*(frame_count +1)]
    original_frames = librosa.util.frame(original_signal, frame_length=frame_length, hop_length=frame_length, axis=0)
    results: dict = sparse_approximation(original_frames, [LD, FD])

    # Plot results
    plt.figure()
    plt.plot(original_signal, label='Original Signal', linestyle='-', color="red")
    first_key = list(results.keys())[0]
    first_value = results[first_key]
    print(f"Original signal max: {max(original_signal)}")
    print(f"First value max: {max(first_value.flatten())}")

    plt.plot(np.concatenate(first_value), label=f'Sparse Approximation for {first_key}', linestyle='--')
    fourth_key = list(results.keys())[5]
    fourth_value = results[fourth_key]
    print(f"Fourth value max: {max(fourth_value.flatten())}")
    plt.plot(np.concatenate(fourth_value), label=f'Sparse Approximation for {fourth_key}', linestyle='-.')

    for (key, value) in results.items():
        print(f"Result {key}:")
        #print("Shape:", value.shape)
        #plt.plot(np.concatenate(value), label=f'Sparse Approximation for {key}', linestyle='--')
    plt.legend()
    plt.title('Sparse Approximation using Fourier Dictionary and OMP')
    plt.show()
