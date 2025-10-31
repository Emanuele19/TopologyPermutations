import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch.optim as optim

from Model import ModelInterface

class GAT(ModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.gat = GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True, dropout=dropout)
        self.bn  = BatchNorm(hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).view(-1)
