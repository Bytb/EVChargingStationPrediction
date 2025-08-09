# import the important libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# pytorch
import torch
import torch.nn as nn
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#-- GRAPH CONVOLUTIONS --
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X, A_sparse):
        # A_sparse is expected to already have self-loops and normalization
        h = torch.sparse.mm(A_sparse, X)
        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, 128)
        self.gcn2 = GCNLayer(128, 128)
        self.gcn3 = GCNLayer(128, out_dim)

    #     return h
    def forward(self, X, A):
        h = self.gcn1(X, A)
        h = nn.functional.relu(h)
        h = nn.functional.dropout(h, p=0.3, training=self.training)
        h = self.gcn2(h, A)
        h = nn.functional.relu(h)
        h = nn.functional.dropout(h, p=0.3, training=self.training)
        h = self.gcn3(h, A)
        return h
