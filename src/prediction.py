import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from itertools import cycle
import datetime as dt
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import Dataset
from matplotlib.pyplot import figure
import warnings
import time
# custom
from mv_lstm import MV_LSTM

warnings.filterwarnings('ignore')

# Constrain pandas output
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('max_columns', 50)
plt.style.use('bmh')

#device (here cpu only)
device = 'cpu'
INPUT_DIR_PATH = '../data/'


