# import the important libraries
# --- PyG GCN model mirroring your custom setup ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, hidden, cached=True)
        self.conv3 = GCNConv(hidden, out_dim, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_weight)
        return h
