# import the important libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import time

# plotting libraries
from matplotlib import pyplot as plt
import seaborn as sns
import tqdm

# pytorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# torch vision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#-- GRAPH CONVOLUTIONS --
"""
    THIS DOES NOT WORK WITH BATCHES YET
"""
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X, A):
        # Step 1: Add self-loops
        A_hat = A + torch.eye(A.size(0), device=A.device)

        # Step 2: Compute D^(-1/2)
        degree = A_hat.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # Avoid NaNs

        # Step 3: Normalize A_hat: D^(-1/2) * A_hat * D^(-1/2)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # Step 4: Propagate and transform
        h = A_norm @ X
        return self.linear(h)
    
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, 64)
        self.gcn2 = GCNLayer(64, 32)
        self.gcn3 = GCNLayer(32, out_dim)


    def forward(self, X, A):
        h = self.gcn1(X, A)
        h = nn.functional.relu(h)
        h = self.gcn2(h, A)
        h = nn.functional.relu(h)
        h = self.gcn3(h, A)

        return h
        #return nn.functional.log_softmax(h, dim = 1)