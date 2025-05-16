
import os
import librosa
import numpy as np
from sklearn.decomposition import DictionaryLearning

class AudioDictionary:
    def __init__(self, dictionary_size, frame_count, train_wav: str = "./dataset/1727.wav"):
        self.train_wav = train_wav
        self.dictionary_size = dictionary_size
        self.frame_count = frame_count

    def learn_dictionary(self, sample_size = 1024):
        """Learn a dictionary from the training features."""
        y, sr = librosa.load(self.train_wav)
        
        frames = librosa.util.frame(y, frame_length=sample_size, hop_length=sample_size//2, axis =0)
        frames = frames[:self.frame_count]
        print(f"The frames shape is {frames.shape}.")
        print("Learning dictionary...")
        dictio = DictionaryLearning(n_components=self.dictionary_size, alpha=1, max_iter=300).fit(frames).components_
        print(f"The dictionary shape is {dictio.shape}.")
        return dictio


# Example usage
if __name__ == "__main__":
    print("Not iplemented yet.")
