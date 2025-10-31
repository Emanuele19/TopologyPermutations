import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, BatchNorm
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch.optim as optim

from .Model import ModelInterface

class APPNPNet(ModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, K=10, alpha=0.1, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1  = BatchNorm(hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.appnp = APPNP(K=K, alpha=alpha, dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.lin1(x)))
        h = self.dropout_layer(h)
        h = self.lin2(h)                  # logits
        out = self.appnp(h, edge_index)   # propagazione
        return out.view(-1)
