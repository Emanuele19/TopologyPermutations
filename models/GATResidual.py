import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from Model import ModelInterface

class GATResidual1L(ModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.gat = GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True, dropout=dropout)
        self.bn  = BatchNorm(hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return (self.head(h) + self.skip(x)).view(-1)