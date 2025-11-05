import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

from .Model import GraphModelInterface

class GCN_Sage(GraphModelInterface):
    """
    GraphSAGE "pulita":
    - ReLU + BatchNorm tra le conv
    - Dropout per regolarizzare
    - NIENTE sigmoid nei layer intermedi (BCEWithLogitsLoss si occupa della sigmoid finale)
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1   = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)  # logits
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
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
