import os
import shutil
import random
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Data Split into Train/Test Folders
def train_test(folder_path, train_ratio=0.7):
    # get file list
    wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    
    random.shuffle(wav_files) # randomize
    train_size = int(train_ratio * len(wav_files))

    train_files = wav_files[:train_size]
    test_files = wav_files[train_size:]
    
    # create train and test folders
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    
    for file in train_files: # move to train folder
        shutil.move(os.path.join(folder_path, file), os.path.join('train', file))
    for file in test_files: # move to test folder
        shutil.move(os.path.join(folder_path, file), os.path.join('test', file))

def time_freq_plots(file_path, emotion):
    signal, sample_rate = librosa.load(file_path)

    # plot audio files in time domain
    plt.figure(1)
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')
    plt.title(emotion)
    plt.show()

    # plot audio files in frequency domain
    k = np.arange(len(signal))
    T = len(signal)/sample_rate
    freq = k/T
    DATA_0 = np.fft.fft(signal)
    abs_DATA_0 = abs(DATA_0)
    plt.figure(2)
    plt.plot(freq, abs_DATA_0)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Amplitude / dB")
    plt.xlim([0, 1000])
    plt.title(emotion)
    plt.show()

    # plot the time-frequency variation of the audio
    D = librosa.stft(signal)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(3)
    librosa.display.specshow(S_db, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title(emotion)
    plt.show()


    # _____DATA SETUP_____
# Run only once: Splits data into seperate folders for 70/30 train/test
# train_test('angry')
# train_test('fear')
# train_test('happy')
# train_test('sad')

# Organize training data in dataframe
file_names_train = []
labels_train = []
for dirname, _, x in os.walk('train'):
    for file in x:
        file_names_train.append(os.path.join(dirname, file))
        label = file.split('_')[-1]
        label = label.split('.')[0]
        labels_train.append(label.lower())

df = pd.DataFrame()
df['Path'] = file_names_train
df['Label'] = labels_train

# Organize training data in dataframe
file_names_test = []
labels_test = []
for dirname, _, x in os.walk('test'):
    for file in x:
        file_names_test.append(os.path.join(dirname, file))
        label = file.split('_')[-1]
        label = label.split('.')[0]
        labels_test.append(label.lower())

df2 = pd.DataFrame()
df2['Path'] = file_names_test
df2['Label'] = labels_test

    # _____DATA EXPLORATION_____
# Number of each label in training data
print(df['Label'].value_counts())

# Time, Frequency Plots for each emotion
# emotions = ["angry", "happy", "fear", "sad"]
# for emotion in emotions:
#     time_freq_plots(df.loc[df['Label'].eq(emotion).idxmax(), 'Path'], emotion)


    # _____FEATURE EXTRACTION & PROCESSING_____
def extract_features(df, data):
    y_emotions = []
    X_features = []
    columns_ = []
    scaler = StandardScaler()
    scaler1 = MinMaxScaler(feature_range=(-1, 1))

    for index, row in df.iterrows():
        signal, sample_rate = librosa.load(row["Path"])

        # Feature: Loudness
        df_loudness = pd.DataFrame()
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S)
        df_loudness['Loudness'] = rms[0]

        loudness_scaled = scaler1.fit_transform(df_loudness) # scale
        average_loud_scaled = np.mean(loudness_scaled, axis=0) # average
        scaled_loud_df = pd.DataFrame([average_loud_scaled])
        scaled_loud_df.columns = df_loudness.columns

        # Feature: MFCC
        df_mfccs = pd.DataFrame()
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)
        for n_mfcc in range(len(mfccs)):
            df_mfccs['MFCC_%d'%(n_mfcc+1)] = mfccs.T[n_mfcc]

        mfcc_scaled = scaler1.fit_transform(df_mfccs) # scale
        average_mfcc_scaled = np.mean(mfcc_scaled, axis=0) # average
        scaled_mfcc_df = pd.DataFrame([average_mfcc_scaled])
        scaled_mfcc_df.columns = df_mfccs.columns

        # Feature: ZCR
        df_zero_crossing_rate = pd.DataFrame()
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        df_zero_crossing_rate['ZCR'] = zcr[0]

        zcr_scaled = scaler1.fit_transform(df_zero_crossing_rate) # scale
        avg_zcr_scaled = np.mean(zcr_scaled, axis=0) # average
        scaled_zcr_df = pd.DataFrame([avg_zcr_scaled])
        scaled_zcr_df.columns = df_zero_crossing_rate.columns

        # Feature: Chroma
        df_chroma = pd.DataFrame()
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
        for n_chroma in range(len(chromagram)):
            df_chroma['Chroma_%d'%(n_chroma+1)] = chromagram.T[n_chroma]

        chroma_scaled = scaler1.fit_transform(df_chroma) # scale
        average_chroma_scaled = np.mean(chroma_scaled, axis=0) # average
        scaled_chroma_df = pd.DataFrame([average_chroma_scaled])
        scaled_chroma_df.columns = df_chroma.columns

        # Feature: Mel Spectogram
        df_mel = pd.DataFrame()
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=12)
        for n_mel in range(len(mel_spectrogram)):
            df_mel['Mel_Spectrogram_%d'%(n_mel+1)] = mel_spectrogram.T[n_mel]

        mel_scaled = scaler1.fit_transform(df_mel) # scale
        average_mel_scaled = np.mean(mel_scaled, axis=0) # average
        scaled_mel_df = pd.DataFrame([average_mel_scaled])
        scaled_mel_df.columns = df_mel.columns

        mayron = pd.concat([df_loudness, df_mfccs, df_zero_crossing_rate, df_chroma, df_mel], axis=1)


        # combine all features
        feature_matrix = pd.concat([scaled_loud_df, scaled_mfcc_df, scaled_zcr_df, scaled_chroma_df, scaled_mel_df], axis=1)
        y_emotions.append(row["Label"])
        if index == 0:
            columns_ = feature_matrix.columns
        X_features.append(feature_matrix.values)

    y_emotions_array = np.array(y_emotions)
    y_train_emotions = pd.DataFrame(y_emotions_array, columns=["Label"])
    y_train_emotions.to_csv(f"y_{data}.csv", index=False)

    X_features_array = np.array(X_features)
    X_features_reshaped = X_features_array.squeeze(axis=1)
    X_train_features = pd.DataFrame(X_features_reshaped, columns=columns_)
    X_train_features.to_csv(f"X_{data}.csv", index=False)

extract_features(df, "train")
extract_features(df2, "test")