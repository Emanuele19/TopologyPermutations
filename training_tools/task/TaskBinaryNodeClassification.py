from __future__ import annotations
from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from training_tools.task.Task import AbstractTask

from models import ModelInterface, SimpleModelInterface, GraphModelInterface


class TaskBinaryNodeClassification(AbstractTask):
    """
    Task per classificazione binaria a livello nodo (BCEWithLogits).

    Modalità supportate:
    - 'transductive' : usa l'intero grafo; loss/metriche sono mascherate da train/val/test mask.
    - 'inductive'    : **subgrafo per fase** (train/val/test) con reindex dei nodi.
                       Evita leakage statistico (norm globali, self-loops, ecc.).

    Loss: BCEWithLogits (supporta `pos_weight`).
    Predizione: sigmoid + threshold.
    Metriche: accuracy, precision, recall, F1 via scikit-learn.

    Convenzioni su `parts` (dallo splitter):
    - 'supervised_mask', 'aux_mask', 'train_pool_mask', 'test_mask', 'cv_fold'.
    """

    def __init__(self,
                 threshold: float = 0.5,
                 pos_weight: Optional[float] = None,
                 reduction: str = "mean"):
        self.threshold = float(threshold)
        self.pos_weight_val = None if pos_weight is None else float(pos_weight)
        self.reduction = reduction

    # ---------------- API ----------------

    def compute_loss(self,
                     model: ModelInterface,
                     data: Data,
                     parts: Dict[str, object],
                     phase: Literal['train', 'val'],
                     fold_k: Optional[int],
                     mode: Literal['transductive', 'inductive'] = "transductive") -> Tensor:
        """
        Calcola la BCEWithLogitsLoss per la fase ('train'|'val') e fold `fold_k`.
        In 'inductive' opera su un subgrafo per fase con reindex dei nodi.
        """
        assert phase in ("train", "val"), "compute_loss: phase deve essere 'train' o 'val'"

        x, y, edge_index, sup_mask = self._select_inputs(data, parts, phase, fold_k, mode)

        # forward
        if isinstance(model, SimpleModelInterface):
            logits = model(x)
        elif isinstance(model, GraphModelInterface):
            logits = model(x, edge_index)
        else:
            try:
                logits = model(x, edge_index)
            except TypeError:
                logits = model(x)

        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.view(-1)

        target = y[sup_mask].to(torch.float32)

        # BCE con (eventuale) pos_weight
        if self.pos_weight_val is None:
            criterion = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)
        else:
            pw = torch.tensor(self.pos_weight_val, dtype=torch.float32, device=logits.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction=self.reduction)

        loss = criterion(logits[sup_mask], target)
        return loss

    def predict(self,
                model: torch.nn.Module,
                data: Data,
                parts: Dict[str, object],
                phase: Literal['test', 'train', 'val'],
                fold_k: Optional[int],
                mode: Literal['transductive', 'inductive'] = "transductive") -> Tuple[Tensor, Tensor]:
        """
        Restituisce (y_true, y_pred) per 'train' | 'val' | 'test'.
        In 'inductive' esegue la predizione sul subgrafo della fase.
        """
        x, y, edge_index, mask = self._select_inputs(data, parts, phase, fold_k, mode)

        if isinstance(model, SimpleModelInterface):
            logits = model(x)
        elif isinstance(model, GraphModelInterface):
            logits = model(x, edge_index)
        else:
            try:
                logits = model(x, edge_index)
            except TypeError:
                logits = model(x)

        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.view(-1)

        prob = torch.sigmoid(logits[mask])
        pred = (prob > self.threshold).to(torch.long)
        y_true = y[mask].to(torch.long)
        return y_true, pred

    def metrics(self, y_true: Tensor, y_pred: Tensor) -> Dict[str, float]:
        """
        Metriche via scikit-learn: accuracy, precision, recall, F1.
        """
        yt = y_true.view(-1).cpu().numpy()
        yp = y_pred.view(-1).cpu().numpy()
        acc = accuracy_score(yt, yp)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # ------------- helper interni --------------

    @staticmethod
    def _fold_masks(parts: Dict[str, object], k: Optional[int]) -> Tuple[Tensor, Tensor]:
        """
        Deriva (train_mask_supervised_k, val_mask_supervised_k) dal dict dello splitter.
        """
        cv_fold: Tensor = parts["cv_fold"]
        train_pool_mask: Tensor = parts["train_pool_mask"]
        supervised_mask: Tensor = parts["supervised_mask"]

        if k is None:
            tr = train_pool_mask & supervised_mask
            va = torch.zeros_like(tr, dtype=torch.bool)
            return tr, va

        tr = (cv_fold != k) & train_pool_mask & supervised_mask
        va = (cv_fold == k) & train_pool_mask & supervised_mask
        return tr, va

    @staticmethod
    def _phase_masks(parts: Dict[str, object],
                     phase: Literal['test', 'train', 'val'],
                     k: Optional[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Restituisce (keep_sup_mask, keep_aux_mask, loss_sup_mask) per la fase richiesta.

        - keep_sup_mask: supervised che appartengono alla fase (train fold / val fold / test).
        - keep_aux_mask: nodi ausiliari (duplicati) da includere come contesto (solo train/val).
        - loss_sup_mask: supervised della fase su cui calcolare loss/metriche.
        """
        supervised_mask: Tensor = parts["supervised_mask"]
        aux_mask: Tensor = parts["aux_mask"]

        if phase == "test":
            keep_sup = parts["test_mask"] & supervised_mask
            keep_aux = torch.zeros_like(aux_mask, dtype=torch.bool)  # niente aux in test
            loss_sup = keep_sup.clone()
            return keep_sup, keep_aux, loss_sup

        tr_sup, va_sup = TaskBinaryNodeClassification._fold_masks(parts, k)
        keep_sup = tr_sup if phase == "train" else va_sup
        keep_aux = aux_mask.clone()  # aux sempre presenti come contesto in train/val
        loss_sup = keep_sup.clone()
        return keep_sup, keep_aux, loss_sup

    def _select_inputs(self,
                       data: Data,
                       parts: Dict[str, object],
                       phase: Literal['test', 'train', 'val'],
                       k: Optional[int],
                       mode: Literal['transductive', 'inductive']
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Restituisce (x, y, edge_index, mask_loss) per la fase/modo richiesti.

        - 'transductive' : (x_all, y_all, edge_index_all, mask_loss_globale)
        - 'inductive'    : (x_sub, y_sub, edge_index_sub, mask_loss_locale_al_subgrafo)
        """
        assert mode in ("transductive", "inductive")

        if mode == "inductive":
            # Subgrafo per fase con reindex dei nodi
            keep_sup, keep_aux, loss_sup = self._phase_masks(parts, phase, k)
            if phase in ("train", "val"):
                keep_nodes = keep_sup | keep_aux
            else:
                keep_nodes = keep_sup  # test: solo supervised

            keep_idx = keep_nodes.nonzero(as_tuple=False).view(-1)
            ei_sub, _ = subgraph(keep_idx, data.edge_index,
                                 relabel_nodes=True, num_nodes=data.num_nodes)
            x_sub = data.x[keep_idx]
            y_sub = data.y[keep_idx]
            loss_mask_sub = loss_sup[keep_idx]
            return x_sub, y_sub, ei_sub, loss_mask_sub

        # transductive: grafo intero + mask globale per la fase
        x_all, y_all = data.x, data.y
        ei = data.edge_index

        if phase == "test":
            mask_loss = parts["test_mask"] & parts["supervised_mask"]
        else:
            tr, va = self._fold_masks(parts, k)
            mask_loss = tr if phase == "train" else va

        return x_all, y_all, ei, mask_loss


    # METODO NON ANCORA USATO
    # Questo metodo serve per implementare gli ego_graph in fase inductive
    # L'obiettivo è valutare le prestazioni del modello addestrato in inductive
    #   su nodi con vicinato invece che singoli
    # QUANDO LO SOSTITUIRAI A QUELLO VECCHIO:
    # - Aggiungi i parametri ego_graph e ego_k ai metodi sopra
    # - Aggiungi i parametri alla chiamata a questa funzione fatta nei
    #   metodi sopra
    # COME TESTARE QUESTO METODO:
    # Il file utils_debug.py plotta il sottografo di test
    # PERCHÉ NON USO ANCORA QUESTO METODO:
    # Al momento il sampling dei nodi per le partizioni di test e training
    #   è completamente casuale, quindi non necessariamente i nodi di test
    #   hanno archi che li collegano, rendendo questo metodo inutile
    # Prima implemento un meccanismo per fare sampling mantenendo il
    #   più possibile la componente connessa e poi potrò usare questo metodo
    def _select_inputs_new(self,
                    data: Data,
                    parts: Dict[str, object],
                    phase: Literal['test', 'train', 'val'],
                    k: Optional[int],
                    mode: Literal['transductive', 'inductive'],
                    ego_graph: bool = False,
                    ego_k: int = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Restituisce (x, y, edge_index, mask_loss) per la fase/modo richiesti.

        - 'transductive' : (x_all, y_all, edge_index_all, mask_globale)
        - 'inductive'    :
            * default: subgrafo per fase con reindex; in test include **solo** supervised (nodi spesso isolati).
            * se `ego_graph=True` e phase='test': batch block-diagonale di ego-graph a r=ego_k
            **limitati alla partizione di test supervised** (no train/val, no aux); la mask seleziona i centri.
        """
        assert mode in ("transductive", "inductive")

        # --- transductive invariato ---
        if mode == "transductive":
            x_all, y_all = data.x, data.y
            ei = data.edge_index
            if phase == "test":
                mask_loss = parts["test_mask"] & parts["supervised_mask"]
            else:
                tr, va = self._fold_masks(parts, k)
                mask_loss = tr if phase == "train" else va
            return x_all, y_all, ei, mask_loss

        # --- inductive ---
        keep_sup, keep_aux, loss_sup = self._phase_masks(parts, phase, k)

        # Caso speciale: TEST + EGO-GRAPH → batch di ego-graph su soli supervised di test
        if phase == "test" and ego_graph:
            device = data.x.device
            keep_test_sup = (parts["test_mask"] & parts["supervised_mask"]).to(torch.bool)
            test_nodes = torch.nonzero(keep_test_sup, as_tuple=False).view(-1)

            # edge_index ristretto ai soli archi intra-test (esclude train/val e aux)
            ei = data.edge_index
            edge_keep_test = keep_test_sup[ei[0]] & keep_test_sup[ei[1]]
            ei_test = ei[:, edge_keep_test]

            # Costruzione batch block-diagonale
            x_chunks, y_chunks = [], []
            ei_chunks = []
            center_masks = []
            offset = 0

            for c in test_nodes.tolist():
                subset, ei_local, mapping, _ = k_hop_subgraph(
                    torch.tensor([c], device=ei_test.device),
                    ego_k,
                    ei_test,
                    relabel_nodes=True,
                    num_nodes=data.num_nodes
                )
                # features/labels locali
                x_sub = data.x[subset]
                y_sub = data.y[subset]

                # reindex per concatenazione block-diagonale
                ei_local = ei_local.clone()
                if ei_local.numel() > 0:
                    ei_local = ei_local + offset

                # center mask locale → mask globale con offset
                cm_local = torch.zeros(subset.numel(), dtype=torch.bool, device=device)
                cm_local[mapping.view(-1)] = True

                # accumula
                x_chunks.append(x_sub)
                y_chunks.append(y_sub)
                ei_chunks.append(ei_local)
                center_masks.append(cm_local)
                offset += subset.numel()

            if len(x_chunks) == 0:
                # Nessun test node (degenera): restituisci tensori vuoti coerenti
                empty = data.x.new_zeros((0, data.x.size(1)))
                return empty, data.y.new_zeros((0,), dtype=data.y.dtype), data.edge_index.new_zeros((2,0)), empty.new_zeros((0,), dtype=torch.bool)

            x_cat = torch.cat(x_chunks, dim=0)
            y_cat = torch.cat(y_chunks, dim=0)
            ei_cat = torch.cat(ei_chunks, dim=1) if ei_chunks and ei_chunks[0].numel() > 0 else data.edge_index.new_zeros((2,0))
            center_mask = torch.cat(center_masks, dim=0)

            return x_cat, y_cat, ei_cat, center_mask

        # Percorso inductive standard (train/val o test senza ego)
        if phase in ("train", "val"):
            keep_nodes = keep_sup | keep_aux
        else:
            keep_nodes = keep_sup  # test: solo supervised

        keep_idx = keep_nodes.nonzero(as_tuple=False).view(-1)
        ei_sub, _ = subgraph(keep_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        x_sub = data.x[keep_idx]
        y_sub = data.y[keep_idx]
        loss_mask_sub = loss_sup[keep_idx]
        return x_sub, y_sub, ei_sub, loss_mask_sub
