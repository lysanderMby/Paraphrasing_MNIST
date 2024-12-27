'''
Training a classifier architecture designed to produce intepretable intermediate states/
These intermediate states are all, until the final logits, of the same shape of the original MNIST inputs.
This harms classification performance (concerningly pointing to a negative alignment tax).

Classification can be performed while the intermediate states are randomly selected to have the paraphraser model applied to them.
Randomly applying this model reduces the potential for steganography in the internal thoughts of the model, although it seems unlikely that this would be an issue at this scale and level of simplicity.

This script shows the impact of randomly applying this paraphrasing. Other scripts show the impact of always or never applying it.
'''

# Package imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18
import os.path
from tqdm import tqdm
import os