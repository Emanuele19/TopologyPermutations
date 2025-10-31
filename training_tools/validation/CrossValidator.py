# src/training/cross_validator.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Literal
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from training_tools.trainer import Trainer
from training_tools.task import AbstractTask


class CrossValidator:
    """
    CrossValidator(trainer, model_factory, mode='transductive', select_by='f1')

    Orchestratore K-fold per single-graph node classification:
    - per ogni fold k costruisce le mask train/val dal dict `parts` (output dello splitter),
    - invoca `trainer.fit(...)` con `task` e `model` "fresh" creato da `model_factory`,
    - tiene traccia del **best model state_dict** (secondo `select_by`) e dello storico per fold,
    - (opz.) esegue la valutazione finale su `test_mask`.

    Parametri
    ----------
    trainer : Trainer
        Oggetto che implementa il loop di addestramento per un fold (train/val).
    model_factory : Callable[[], nn.Module]
        Funzione/factory che restituisce un **nuovo** modello inizializzato a ogni fold.
        Esempio: `lambda: MyGNN(in_channels=d, hidden=128, ...)`.
    mode : {'transductive','inductive'}
        Modalità di utilizzo del grafo. In induttivo, il `task` selezionerà edge mask per fold/fase.
    select_by : {'f1','acc','rec','prec'}
        Metica di selezione per il **miglior fold** (default: F1-val).

    API
    ---
    run(task, data, parts) -> (artifacts, summary)
        Esegue i K fold e ritorna:
          - artifacts: dict con 'per_fold' (history metriche) e 'best_state'
          - summary:   dict con statistiche aggregate sulle metriche val (mean/std) e info miglior fold

    test(task, data, parts, state_dict) -> metrics_test
        Valuta un modello sul test set usando lo `state_dict` passato.

    Requisiti su `parts` (dallo splitter)
    --------------------------------------
    - 'cv_fold'           : LongTensor [N] con valori in {0..K-1} per supervised in train_pool, -1 altrimenti
    - 'train_pool_mask'   : BoolTensor [N]
    - 'supervised_mask'   : BoolTensor [N]
    - 'test_mask'         : BoolTensor [N]
    - (induttivo) 'edge_keep_mask_train': List[BoolTensor [E]] per fold k
    - (induttivo) 'edge_keep_mask_val'  : List[BoolTensor [E]] per fold k
    """

    def __init__(self,
                 trainer: Trainer,
                 model_factory: Callable[[], nn.Module],
                 mode: Literal['inductive', 'transductive'] = "transductive",
                 select_by: Literal['accuracy', 'precision', 'recall', 'f1'] = "f1"):
        assert mode in ("transductive", "inductive")
        assert select_by in ("f1", "accuracy", "recall", "precision")
        self.trainer = trainer
        self.model_factory = model_factory
        self.mode = mode
        self.select_by = select_by

    # ------------------- public -------------------

    def run(self,
            task: AbstractTask,
            data: Data,
            parts: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
        """
        Esegue K fold definiti in `parts['cv_fold']`.

        Ritorna
        -------
        artifacts : dict
        - 'per_fold' : List[dict] con storici per fold (history del trainer + metriche finali val)
        - 'best_state': dict (state_dict del modello con miglior metrica val)
        - 'best_fold': int (indice del fold migliore)
        summary : dict
        - 'val_mean' : dict con media delle metriche val
        - 'val_std'  : dict con std delle metriche val
        - 'select_by': metrica usata per selezione best
        """
        cv_fold: torch.Tensor = parts["cv_fold"]
        K = int(cv_fold.max().item() + 1) if cv_fold.numel() > 0 else 1

        per_fold: List[Dict[str, object]] = []
        best_state: Optional[Dict[str, Tensor]] = None
        best_score: float = -float("inf")
        best_fold: int = -1

        vals_acc, vals_prec, vals_rec, vals_f1 = [], [], [], []

        for k in range(K):
            model = self.model_factory()
            # addestra per il fold k
            best_state_k, history = self.trainer.fit(
                model=model, task=task, data=data, parts=parts, fold_k=k, mode=self.mode
            )

            # metriche finali su VAL nel best epoch (le ricalcoliamo pulite)
            model.load_state_dict(best_state_k)
            model.eval()
            with torch.no_grad():
                y_val, yhat_val = task.predict(model, data, parts, phase="val", fold_k=k, mode=self.mode)
                m_val = task.metrics(y_val, yhat_val)

            vals_acc.append(m_val["accuracy"])
            vals_prec.append(m_val["precision"])
            vals_rec.append(m_val["recall"])
            vals_f1.append(m_val["f1"])

            per_fold.append({
                "fold": k,
                "fold": k,
                "history": history,     # curva per epoca
                "val_metrics": m_val,   # metriche calcolate con best_state_k
            })

            score = m_val[self.select_by]
            if score > best_score:
                best_score = score
                best_state = best_state_k
                best_fold = k

        # aggregati su validation
        val_mean = {
            "accuracy": float(np.mean(vals_acc) if vals_acc else float("nan")),
            "precision": float(np.mean(vals_prec) if vals_prec else float("nan")),
            "recall": float(np.mean(vals_rec) if vals_rec else float("nan")),
            "f1": float(np.mean(vals_f1) if vals_f1 else float("nan")),
        }
        val_std = {
            "accuracy": float(np.std(vals_acc) if vals_acc else float("nan")),
            "precision": float(np.std(vals_prec) if vals_prec else float("nan")),
            "recall": float(np.std(vals_rec) if vals_rec else float("nan")),
            "f1": float(np.std(vals_f1) if vals_f1 else float("nan")),
        }

        artifacts = {
            "per_fold": per_fold,
            "best_state": best_state,
            "best_fold": best_fold,
        }
        summary = {
            "val_mean": val_mean,
            "val_std": val_std,
            "select_by": self.select_by,
        }
        return artifacts, summary

    def test(self,
             task: AbstractTask,
             data: Data,
             parts: Dict[str, object],
             state_dict: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Valuta sul test set usando `state_dict` (tipicamente quello migliore sui fold).
        """
        model = self.model_factory()
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            y_te, yhat_te = task.predict(model, data, parts, phase="test", fold_k=None, mode=self.mode)
            m_te = task.metrics(y_te, yhat_te)
        return m_te
