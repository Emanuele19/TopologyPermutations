# src/training/trainer.py
from __future__ import annotations
from typing import Dict, Optional, List, Tuple, Literal
import math
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from training_tools.task import Task


class Trainer:
    """
    Trainer(lr=1e-3, weight_decay=5e-4, max_epochs=50, patience=10,
            device='auto', grad_clip_norm=None, use_grad_scaler=False, log_every=1)

    Esegue il ciclo di addestramento per **un fold** (train/val), con:
      - ottimizzazione Adam,
      - early stopping su **F1-val**,
      - checkpoint in memoria dello **stato migliore**,
      - supporto transduttivo/induttivo (selezione edge_index demandata al Task).

    Parametri
    ----------
    lr : float
        Learning rate per Adam.
    weight_decay : float
        L2 weight decay per Adam.
    max_epochs : int
        Epoche massime.
    patience : int
        Numero di epoche senza miglioramento F1-val prima di fermarsi.
    device : {'auto','cpu','cuda'} | torch.device
        Dispositivo di esecuzione. 'auto' → cuda se disponibile, altrimenti cpu.
    grad_clip_norm : Optional[float]
        Se impostato, applica gradient clipping (norma globale) a ogni step.
    use_grad_scaler : bool
        Se True, usa autocast/GradScaler (mixed precision) in forward/backward.
    log_every : int
        Frequenza di logging (epoche). Se 0, nessun log.

    API
    ---
    fit(model, task, data, parts, fold_k, mode='transductive') -> (best_state_dict, history)
        Esegue train/val per il fold `fold_k`. Ritorna lo **state_dict** migliore su val
        e uno storico con metriche per epoca e best summary.

    Requisiti
    ---------
    - `task` implementa:
        * compute_loss(model, data, parts, phase, fold_k, mode) -> Tensor (scalare)
        * predict(model, data, parts, phase, fold_k, mode) -> (y_true, y_pred)
        * metrics(y_true, y_pred) -> dict con 'f1' almeno
    - `parts` è il dict dello splitter (maschere e, se induttivo, edge mask per fold).
    """

    def __init__(self,
                 lr: float = 1e-3,
                 weight_decay: float = 5e-4,
                 max_epochs: int = 50,
                 patience: int = 10,
                 device: str | torch.device = "auto",
                 grad_clip_norm: Optional[float] = None,
                 use_grad_scaler: bool = False,
                 log_every: int = 0):
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip_norm = grad_clip_norm
        self.use_grad_scaler = bool(use_grad_scaler)
        self.log_every = int(log_every)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._scaler = torch.amp.GradScaler('cuda', enabled=self.use_grad_scaler)

    # ------------------- public -------------------

    def fit(self,
            model: nn.Module,
            task: Task,
            data: Data,
            parts: Dict[str, object],
            fold_k: Optional[int],
            mode: Literal['inductive', 'transductive'] = "transductive") -> Tuple[Dict[str, Tensor], Dict[str, object]]:
        """
        Addestra `model` per un fold `fold_k` usando `task`, su `data` e `parts` dello splitter.

        Parametri
        ----------
        model : nn.Module
            Modello PyTorch (solo __init__/forward).
        task : TaskBinaryNodeClassification (o compatibile)
            Politica di loss/pred/metriche e selezione edge_index (transductivo/induttivo).
        data : Data
            Grafo PyG.
        parts : dict
            Output dello splitter (maschere e, se induttivo, edge mask per fold).
        fold_k : Optional[int]
            Indice del fold corrente (0..K-1). Puoi passare None se non usi K-fold.
        mode : {'transductive','inductive'}
            Modalità di grafo per train/val.

        Ritorna
        -------
        best_state_dict : dict
            Parametri del modello con **miglior F1-val**.
        history : dict
            - 'epochs_run': epoche effettivamente eseguite
            - 'best_epoch': epoca con F1-val migliore
            - 'best_val_f1': valore F1 in best_epoch
            - 'train': List[dict] per epoca (loss, f1, ecc.)
            - 'val'  : List[dict] per epoca (loss, f1, ecc.)
        """
        model = model.to(self.device)
        data = self._to_device_data(data)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state: Dict[str, Tensor] = {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
        best_val_f1: float = -math.inf
        best_epoch: int = -1
        patience_ctr = 0

        hist_train: List[Dict[str, float]] = []
        hist_val: List[Dict[str, float]] = []

        for epoch in range(1, self.max_epochs + 1):
            # ---- TRAIN
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_grad_scaler): # se enabled esegui con mixed precision
                loss = task.compute_loss(model, data, parts, phase="train", fold_k=fold_k, mode=mode)

            self._scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

            self._scaler.step(optimizer)
            self._scaler.update()


            # metriche train (su supervised train del fold)
            with torch.no_grad():
                y_tr, yhat_tr = task.predict(model.eval(), data, parts, phase="train", fold_k=fold_k, mode=mode)
                m_tr = task.metrics(y_tr, yhat_tr)
                m_tr["loss"] = float(loss.detach().cpu())
                hist_train.append(m_tr)

            # ---- VAL
            model.eval()
            with torch.no_grad():
                # loss val (calcolata come train ma su val_mask)
                loss_val = task.compute_loss(model, data, parts, phase="val", fold_k=fold_k, mode=mode)
                y_va, yhat_va = task.predict(model, data, parts, phase="val", fold_k=fold_k, mode=mode)
                m_va = task.metrics(y_va, yhat_va)
                m_va["loss"] = float(loss_val.detach().cpu())
                hist_val.append(m_va)

                val_f1 = float(m_va["f1"])

            # ---- Early stopping su F1-val
            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if self.log_every and (epoch % self.log_every == 0):
                print(f"[Epoch {epoch:03d}] "
                      f"train: loss={m_tr['loss']:.4f} f1={m_tr['f1']:.3f} | "
                      f"val: loss={m_va['loss']:.4f} f1={m_va['f1']:.3f} "
                      f"{'(*)' if improved else ''}")

            if patience_ctr >= self.patience:
                if self.log_every:
                    print(f"Early stopping (no val F1 improvement for {self.patience} epochs).")
                break

        history = {
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1 if best_epoch != -1 else float('nan'),
            "train": hist_train,
            "val": hist_val,
        }
        return best_state, history

    # ------------------- helpers -------------------

    def _to_device_data(self, data: Data) -> Data:
        """
        Sposta tensori di `data` sul device del trainer (x, y, edge_index, edge_attr, mask note).
        Mantiene liste python (es. string_id) intatte.
        """
        out = data
        for key, val in list(data):
            if isinstance(val, torch.Tensor):
                setattr(out, key, val.to(self.device))
        return out
