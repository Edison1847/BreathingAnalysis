import os
import matplotlib.pyplot as plt
import noisereduce as nr
import librosa
import numpy as np
from pydub import AudioSegment
from PIL import Image
import librosa.display
import numpy as np
import library as lib
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))



pathIn = path +'/raw_sound_files/'
pathOut = path+'/normalized/'
imgPathOut = path+'/processed_files/'
# low_freq =
# high_fre =


audioFileList = lib.fn(pathIn)
# lib.saveBandpassNormalized(audioFileList, pathIn, pathOut, imgPathOut)
lib.saveImgGraph(audioFileList, pathIn, pathOut, imgPathOut)