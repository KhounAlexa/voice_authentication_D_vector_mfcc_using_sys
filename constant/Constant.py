import sys
import torch
import warnings
import os
import librosa
import pandas
import pyrubberband
import soundfile as sf
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import h5py
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from audiomentations import Compose, TimeMask, Gain
from audiomentations.core.transforms_interface import BaseWaveformTransform
import random
import time

# Paths to your data

TRAIN_CSV = './labels/train_labels.csv'  # Path to the training DataFrame CSV
TEST_CSV = './labels/test_labels.csv'    # Path to the test DataFrame CSV

# Audio Processing
N_MFCC = 40

# Model Parameters
INPUT_DIM = N_MFCC
EMBEDDING_DIM = 128
# abcde
# Training Parameters
BATCH_SIZE = 64
user_name = sys.argv[1]
USER = user_name
# 3e-5
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
LOAD = 10
WEIGHT_DECAY = 1e-5
FRAC = 1
# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Authentication Parameters
THRESHOLD = 0.45  # Adjust based on validation

# Suppress all warnings
warnings.filterwarnings("ignore")
