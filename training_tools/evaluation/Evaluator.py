# src/training/evaluator.py
from __future__ import annotations
from typing import Callable, Dict, Literal
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from training_tools.task import AbstractTask


class Evaluator:
    """
    Evaluator(model_factory, mode='transductive', with_confusion=False, with_report=False)

    Valuta un modello su un grafo singolo usando le maschere prodotte dallo splitter.
    È agnostico al tipo di Task, purché implementi l'interfaccia BaseNodeTask.

    Parametri
    ----------
    model_factory : Callable[[], nn.Module]
        Factory che crea un **nuovo** modello (stessa architettura dei fold).
    mode : {'transductive','inductive'}
        Modalità del grafo per la predizione ('inductive' usa le edge mask del Task).
    with_confusion : bool
        Se True, calcola e ritorna anche la confusion matrix su test.
    with_report : bool
        Se True, ritorna anche lo sklearn classification report (stringa).

    Metodi
    ------
    evaluate(task, data, parts, state_dict) -> Dict[str, object]
        Valuta sul **test set** (parts['test_mask']) e ritorna metriche + (opz.) confusion/report.
    evaluate_on_mask(task, data, mask, state_dict, phase='custom') -> Dict[str, float]
        Valuta su una maschera arbitraria (utile per analisi ad-hoc).
    """

    def __init__(self, model_factory: Callable[[], nn.Module], 
                 mode: Literal['inductive', 'transductive'] = "transductive") -> None:
        assert mode in ("transductive", "inductive")
        self.model_factory = model_factory
        self.mode = mode

    @torch.no_grad()
    def evaluate(self, 
                 task: AbstractTask, 
                 data: Data, 
                 parts: Dict[str, object], 
                 state_dict: Dict[str, Tensor],
                 with_confusion: bool = False,
                 with_report: bool = False) -> Dict[str, object]:
        """
        Valuta sul **test set** definito dallo splitter: parts['test_mask'] & parts['supervised_mask'].
        Ritorna un dict con metriche sklearn ('accuracy','precision','recall','f1') e, opzionalmente,
        'confusion_matrix' (np.ndarray 2x2) e 'classification_report' (str).
        """
        model = self.model_factory()
        model.load_state_dict(state_dict)
        model.eval()

        y_true, y_pred = task.predict(model, data, parts, phase="test", fold_k=None, mode=self.mode)
        yt = y_true.view(-1).cpu().numpy()
        yp = y_pred.view(-1).cpu().numpy()

        acc = accuracy_score(yt, yp)
        prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)

        out: Dict[str, object] = {
            "n_test": int(yt.size),
            "class_counts_test": {int(c): int((yt == c).sum()) for c in np.unique(yt)},
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        if with_confusion:
            cm = confusion_matrix(yt, yp, labels=[0,1])
            out["confusion_matrix"] = cm
        if with_report:
            out["classification_report"] = classification_report(yt, yp, digits=3, zero_division=0)

        return out

    @torch.no_grad()
    def evaluate_on_mask(self, task: AbstractTask, data: Data, mask: torch.Tensor, state_dict: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Valuta su una **maschera arbitraria** di nodi (bool [N]).
        Utile per analisi mirate (es. solo AD, solo PD, solo un sottogruppo).
        """
        model = self.model_factory()
        model.load_state_dict(state_dict)
        model.eval()

        # Costruisce un "parts" minimale con la sola mask richiesta.
        parts = {
            "supervised_mask":  torch.as_tensor(mask, dtype=torch.bool, device=data.y.device),
            "test_mask":        torch.as_tensor(mask, dtype=torch.bool, device=data.y.device),
        }
        y_true, y_pred = task.predict(model, data, parts, phase="test", fold_k=None, mode=self.mode)
        yt = y_true.view(-1).cpu().numpy()
        yp = y_pred.view(-1).cpu().numpy()

        acc = accuracy_score(yt, yp)
        prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        return {
            "accuracy": float(acc), 
            "precision": float(prec), 
            "recall": float(rec), 
            "f1": float(f1)}
