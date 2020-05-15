import matplotlib.pyplot as plt
import IPython.display
from ipywidgets import interact, interactive, fixed

# Packages
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import library as lib
import os
import sys

def fn(path):
    file_list=os.listdir(path)
    return file_list

def saveBandpassNormalized(audioFileList, pathIn, pathOut, imgPathOut):
    if(0 < len(audioFileList)):
        for x in range(len(audioFileList)):
            s1 = Sound(audioFileList[x], pathIn, pathOut, imgPathOut)
            s1.normalize()

def saveImgGraph(audioFileList, pathIn, pathOut, imgPathOut):
    if (0 < len(audioFileList)):
        for x in range(len(audioFileList)):
            s2 = lib.Sound(audioFileList[x], pathIn, pathOut, imgPathOut)
            s2.saveImg()




def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def prepro(file_name):
    audio_data, sampling_rate = librosa.load(file_name)
    noisy_part = audio_data[0:20000]
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    librosa.output.write_wav("_dir/" + file_name.split('/')[1], reduced_noise, sampling_rate)
    sound = AudioSegment.from_wav("_dir/" + file_name.split('/')[1])
    x = sound.low_pass_filter(700)
    x = x.high_pass_filter(100)
    x.export("_dir/" + file_name.split('/')[1], format="wav")
    sound_filtered = AudioSegment.from_file("_dir/" + file_name.split('/')[1])
    normalized_sound = match_target_amplitude(sound_filtered, -20)
    normalized_sound.export("processed/" + file_name.split('/')[1], format="wav")





class Sound:
    def __init__(self, name, pathIn, pathOut, imgPathOut, lowPassFreq = 2000, highPassFreq = 150):
        self.fileName = name
        self.pathIn = pathIn
        self.pathOut = pathOut
        self.imagePathOut = imgPathOut
        self.lowPassFrequency = lowPassFreq
        self.highPassFrequency = highPassFreq

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)



    def normalize(self):
        soundFile = AudioSegment.from_file(self.pathIn + self.fileName)
        soundFileFilteredL = soundFile.low_pass_filter(self.lowPassFrequency)
        soundFileFilteredLH = soundFileFilteredL.high_pass_filter(self.highPassFrequency)
        soundFileFilteredLH.export(self.pathOut+self.fileName, format="wav")
        soundFile = AudioSegment.from_file(self.pathOut + self.fileName)
        os.remove(self.pathOut + self.fileName)
        normalized_sound = self.match_target_amplitude(soundFile, -20)
        normalized_sound.export(self.pathOut+self.fileName, format="wav")

    def saveImg(self):
        data, rate = librosa.load(self.pathOut + self.fileName)
        data = np.multiply(data, 150)
        fig = plt.figure(figsize=(25, 10))
        ax1 = fig.subplots()  # Creates the Axes object to display one of the plots
        ax2 = ax1.twinx()  # Creates a second Axes object that shares the x-axis
        spec = np.abs(librosa.stft(data, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        ax1 = librosa.display.specshow(spec, sr=rate, x_axis='time', y_axis='log')
        librosa.display.waveplot(data, sr=rate, ax=ax2, alpha=1, color='black')
        fig.tight_layout()
        # ax2.set_ylim([-250, 250])
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fileName = self.fileName
        fileName = fileName[:-4]
        fig.savefig(self.imagePathOut+fileName, transparent=True)
        # plt.show()

