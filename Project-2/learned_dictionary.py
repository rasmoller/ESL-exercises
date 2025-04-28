# Obviously all this code is from Le Chat. 
# I could not find any examples or guides using any libraries to do learned dictionaries on audio files.

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import DictionaryLearning

class AudioDictionaryLearner:
    def __init__(self, train_wav_dir: str, train_csv_dir: str, test_wav_dir: str, test_csv_dir: str):
        self.train_wav_dir = train_wav_dir
        self.train_csv_dir = train_csv_dir
        self.test_wav_dir = test_wav_dir
        self.test_csv_dir = test_csv_dir
        self.train_features = []
        self.train_labels = []
        self.test_features = []
        self.test_labels = []

    def load_and_extract_features(self, wav_dir, csv_dir, is_train=True):
        """Load audio files and extract features along with metadata."""
        for filename in os.listdir(wav_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(wav_dir, filename)
                y, sr = librosa.load(file_path, sr=None)

                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs.T, axis=0)

                # Load corresponding CSV file
                csv_filename = filename.replace('.wav', '.csv')
                csv_path = os.path.join(csv_dir, csv_filename)
                metadata = pd.read_csv(csv_path)

                # Combine features and metadata (example: using start_time and note)
                combined_features = np.concatenate((mfccs_mean, metadata[['start_time', 'note']].mean().values))

                if is_train:
                    self.train_features.append(combined_features)
                else:
                    self.test_features.append(combined_features)

        # Convert lists to NumPy arrays
        if is_train:
            self.train_features = np.array(self.train_features)
        else:
            self.test_features = np.array(self.test_features)

    def learn_dictionary(self, n_components=30, n_iter=100):
        """Learn a dictionary from the training features."""
        dictionary_learning = DictionaryLearning(n_components=n_components, max_iter=n_iter)
        dictionary = dictionary_learning.fit(self.train_features).components_
        print("Dictionary learned successfully!")
        return dictionary

# Example usage
if __name__ == "__main__":
    train_wav_dir = 'Project-2/dataset/musicnet/train_data/' # Project-2\dataset\musicnet\train_data
    train_csv_dir = 'Project-2/dataset/musicnet/train_labels/'
    test_wav_dir = 'Project-2/dataset/musicnet/test_data/'
    test_csv_dir = 'Project-2/dataset/musicnet/test_labels/'

    learner = AudioDictionaryLearner(train_wav_dir, train_csv_dir, test_wav_dir, test_csv_dir)

    # Load and extract features for training data
    learner.load_and_extract_features(train_wav_dir, train_csv_dir, is_train=True)

    # Learn a dictionary from the training features
    dictionary = learner.learn_dictionary()
    print("Learned Dictionary Shape:", dictionary.shape)

    # Load and extract features for test data
    learner.load_and_extract_features(test_wav_dir, test_csv_dir, is_train=False)
