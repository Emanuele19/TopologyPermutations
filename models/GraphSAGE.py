from .Model import GraphModelInterface

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm  # BatchNorm1d per nodi

from typing import Literal

class GCN_Sage(GraphModelInterface):
    """
    GraphSAGE flessibile per node classification (induttivo)
    - num_layers: numero di SAGEConv (k = num_layers)
    - head: 'linear' (default) o 'mlp'
    - use_bn: BatchNorm tra le conv (e nella head MLP)
    - Nessuna sigmoid nel modello: usa BCEWithLogitsLoss (binaria) o CrossEntropyLoss (multiclasse)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, num_layers: int = 2,
        dropout: float = 0.2, project: bool = False, use_bn: bool = True, head_type: Literal['linear', 'mlp'] = 'linear'
    ):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        assert num_layers >= 1, "num_layers deve essere >= 1"
        assert head_type in {"linear", "mlp"}

        self.use_bn = use_bn

        # --- Stack di SAGEConv ---
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Primo layer: in -> hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels, project=project))
        self.bns.append(BatchNorm(hidden_channels) if use_bn else nn.Identity())

        # Layer 2..L: hidden -> hidden
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, project=project))
            self.bns.append(BatchNorm(hidden_channels) if use_bn else nn.Identity())

        self.drop = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        # --- Head di classificazione ---
        head_in = hidden_channels
        if head_type == "linear":
            self.head = nn.Linear(head_in, out_channels)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.BatchNorm1d(head_in) if use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(head_in, out_channels),
            )

    def forward(self, x, edge_index):
        """
        Ritorna:
          - shape (N,) se out_channels == 1 (binaria, per BCEWithLogitsLoss)
          - shape (N, C) se out_channels > 1 (multiclasse, per CrossEntropyLoss)
        """
        h = x

        for _, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)

            h_new = F.relu(h_new)
            h_new = self.drop(h_new)
            h = h_new  # aggiornamento

        logits = self.head(h)
        if self.out_channels == 1:
            return logits.view(-1)
        return logits

