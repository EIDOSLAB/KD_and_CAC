#pag.26

#%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

# require python >= 3.6
# install dependencies using (no GPU):
# python -m pip install torch torchvision torchaudio
# see pytorch.com/get-started/locally for other type of installations (with CUDA, using conda, ...)
