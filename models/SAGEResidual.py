import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch.optim as optim

from models.Model import GraphModelInterface


class SAGEResidual1L(GraphModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.1):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)

        self.conv = SAGEConv(in_channels, hidden_channels)
        self.bn   = BatchNorm(hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
        # skip per allineare le dimensioni all’uscita
        self.skip = nn.Linear(in_channels, out_channels, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout_layer(h)
        out = self.head(h) + self.skip(x)   # residuo sull’output
        return out.view(-1)                 # logits per BCEWithLogitsLoss
