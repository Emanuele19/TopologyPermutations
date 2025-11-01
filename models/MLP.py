import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.optim as optim
from .Model import SimpleModelInterface

class MLP(SimpleModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1)  # logits per BCEWithLogitsLosss