import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle

import torch as T
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader

from Src.dataloader import *
from Src.model import *
from Src.tools import *
from Src.train import *
from Src.test import *

data, loader = get_loader()

G = vsumG(feat_size=2048, hid_size=256, max_out_len=data.max_out_len).cuda()
D = vsumD().cuda()

train_G(G, data, loader, save_func=save_model)
train_GD(G, D, loader, save_func=save_model)

G.load_state_dict(T.load('Model/G.pt'))
D.load_state_dict(T.load('Model/D.pt'))

f1 = evaluation(G, loader)

print('F1: %.4f'%(f1))
