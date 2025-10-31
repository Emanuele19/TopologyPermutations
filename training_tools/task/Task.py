from __future__ import annotations
from typing import Dict, Optional, Tuple, Literal
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch_geometric.data import Data

class AbstractTask(ABC):
    """
    Interfaccia per task di classificazione a livello nodo (single-graph).

    Il Trainer richiede che ogni Task concreto implementi:
      - compute_loss(model, data, parts, phase, fold_k, mode) -> Tensor (scalare)
      - predict(model, data, parts, phase, fold_k, mode) -> (y_true: Tensor, y_pred: Tensor)
      - metrics(y_true, y_pred) -> dict con almeno la chiave 'f1' (float)

    Convenzioni:
      - `phase` ∈ {'train','val','test'}
      - `fold_k` è l'indice del fold corrente (o None se non usi K-fold).
      - `mode` ∈ {'transductive','inductive'} seleziona l'edge_index da usare.

    Note:
      - `parts` è il dict prodotto dallo splitter, contenente maschere per nodi
        e, in modalità induttiva, maschere per archi per fold/fase.
      - I Task possono gestire BCE/CE/AUC ecc.; il Trainer resta agnostico.
    """

    @abstractmethod
    def compute_loss(self,
                     model: torch.nn.Module,
                     data: Data,
                     parts: Dict[str, object],
                     phase: Literal['test', 'train', 'val'],
                     fold_k: Optional[int],
                     mode: Literal['inductive', 'transductive'] = "transductive") -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(self,
                model: torch.nn.Module,
                data: Data,
                parts: Dict[str, object],
                phase: Literal['test', 'train', 'val'],
                fold_k: Optional[int],
                mode: Literal['inductive', 'transductive'] = "transductive") -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def metrics(self, y_true: Tensor, y_pred: Tensor) -> Dict[str, float]:
        raise NotImplementedError
