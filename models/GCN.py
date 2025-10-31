import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.optim as optim
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, add_self_loops


from .Model import GraphModelInterface

class GCN(GraphModelInterface):
    """
    GCN "pulita":
    - Self-loops aggiunti esplicitamente (una sola volta per forward)
    - ReLU + BatchNorm tra le conv
    - Dropout
    - Nessuna sigmoid interna
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        # Imposta add_self_loops=False: li gestiamo noi nel forward per coerenza
        self.conv1 = GCNConv(in_channels,  hidden_channels, add_self_loops=False, normalize=True)
        self.bn1   = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=True)
        self.bn2   = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels,  add_self_loops=False, normalize=True)  # logits
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        # Aggiungi self-loops una sola volta qui (coerente per tutti i layer)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Layer 3 (logits, no activation)
        x = self.conv3(x, edge_index)
        return x.view(-1)  # logits per BCEWithLogitsLoss
