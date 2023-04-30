import cv2
import os
import numpy as np
from keras.utils.image_utils import img_to_array
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras import backend as k
from keras.callbacks import EarlyStopping

