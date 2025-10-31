from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict, Literal
import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import BaseTransform


class MergeRelationsToHomogeneous(BaseTransform):
    """
    >>> MergeRelationsToHomogeneous(merge=('AD','PD'), node_type='protein',
                               label_attr='y',
                               edge_attr_key='edge_attr',
                               edge_attr_reduce='none',
                               add_edge_type=True)

    Converte un `HeteroData` multi-relazione (es. edge types 'AD' e 'PD' tra 'protein')
    in un grafo **omogeneo** `Data` con:
      - `x` e `y` copiati da `data[node_type]`
      - `edge_index` ottenuto **concatenando** gli archi delle relazioni indicate
      - (opzionale) `edge_attr` trattato secondo una strategia di riduzione
      - (opzionale) `edge_type` (intero) che indica, per ogni arco, da quale relazione proviene

    ----------
    Parametri
    - merge: Iterable[str]
        Lista dei nomi di relazione da fondere (default: ('AD','PD')).
    - node_type: str
        Tipo di nodo in `HeteroData` da utilizzare (default: 'protein').
    - label_attr: str
        Nome del campo etichetta nei nodi (default: 'y').
    - edge_attr_key: str
        Nome del tensore di pesi/attributi sugli archi nelle relazioni (default: 'edge_attr').
        Se non presente, i pesi vengono considerati assenti.
    - edge_attr_reduce: {'none', 'stack', 'sum', 'mean'}
        Strategia per trattare i pesi quando si uniscono le relazioni.
    - add_edge_type: bool
        Se True, aggiunge `edge_type: LongTensor [E]` con ID di relazione (0..len(merge)-1).

    Output
    - Ritorna un oggetto `torch_geometric.data.Data` con:
        x: [N, F], y: [N], edge_index: [2, E], (opz.) edge_attr: [E, 1], (opz.) edge_type: [E]
      Mantiene eventuali maschere per-nodo (train/val/test_mask) se presenti su `node_type`.

    Requisiti
    - `data` è un `HeteroData` con edge types (node_type, rel, node_type) per ogni `rel in merge`.
    - Se `edge_attr_reduce` è 'sum' o 'mean', la deduplicazione considera **esattamente** la stessa
      coppia (u,v); il grafo è trattato come orientato. Se vuoi coalescere come **non orientato**,
      prepara prima il grafo con `ToUndirected()`.
    """

    def __init__(self,
                 merge: Iterable[str] = ('AD', 'PD'),
                 node_type: str = 'protein',
                 label_attr: str = 'y',
                 edge_attr_key: str = 'edge_attr',
                 edge_attr_reduce: str = Literal['none', 'stack', 'sum', 'mean'],
                 add_edge_type: bool = True,):
        self.merge = tuple(merge)
        self.node_type = node_type
        self.label_attr = label_attr
        self.edge_attr_key = edge_attr_key
        assert edge_attr_reduce in ('none', 'stack', 'sum', 'mean')
        self.edge_attr_reduce = edge_attr_reduce
        self.add_edge_type = add_edge_type

    def __call__(self, data: HeteroData) -> Data:
        node_store = data[self.node_type]
        x = node_store.x
        y = getattr(node_store, self.label_attr) if hasattr(node_store, self.label_attr) else None

        # Raccogli edge_index (e pesi, se presenti) per ciascuna relazione
        edge_indices: List[Tensor] = []
        edge_weights: List[Optional[Tensor]] = []
        edge_types: List[Tensor] = []

        for ridx, rel in enumerate(self.merge):
            key = (self.node_type, rel, self.node_type)
            ei = data[key].edge_index
            edge_indices.append(ei)

            if hasattr(data[key], self.edge_attr_key):
                w = getattr(data[key], self.edge_attr_key)
                w = w.reshape(-1, 1).to(ei.device)
            else:
                w = None
            edge_weights.append(w)

            if self.add_edge_type:
                edge_types.append(torch.full((ei.size(1),), ridx, dtype=torch.long, device=ei.device))

        # Concatenazione base
        edge_index = torch.cat(edge_indices, dim=1) if len(edge_indices) > 1 else edge_indices[0]
        etype = torch.cat(edge_types, dim=0) if (self.add_edge_type and len(edge_types) > 0) else None

        # Trattamento edge_attr
        if self.edge_attr_reduce == 'none':
            edge_attr = None
        elif self.edge_attr_reduce == 'stack':
            # Mantieni i pesi originali per arco (una colonna), semplicemente concatenati in stesso ordine degli edge
            ws = [w for w in edge_weights if w is not None]
            edge_attr = torch.cat(ws, dim=0) if len(ws) > 0 else None
        else:
            # 'sum' o 'mean': deduplica per (u,v) aggregando i pesi (se assenti, usa 1.0 come default)
            # Implementazione semplice su CPU (robusta senza torch-sparse)
            ei_cpu = edge_index.cpu()
            if any(w is None for w in edge_weights):
                # se qualche relazione non ha pesi, diamo 1.0 a quegli archi
                weights_list = []
                cursor = 0
                for ei_rel, w in zip(edge_indices, edge_weights):
                    e_rel = ei_rel.size(1)
                    if w is None:
                        weights_list.append(torch.ones((e_rel, 1), dtype=torch.float32))
                    else:
                        weights_list.append(w.cpu().to(torch.float32))
                    cursor += e_rel
                w_all = torch.cat(weights_list, dim=0)
            else:
                w_all = torch.cat([w.cpu().to(torch.float32) for w in edge_weights], dim=0)

            # hash semplice degli edge (u,v) come tuple -> aggrega
            src = ei_cpu[0].tolist()
            dst = ei_cpu[1].tolist()
            agg: Dict[Tuple[int, int], List[float]] = {}
            for (u, v), w in zip(zip(src, dst), w_all.tolist()):
                agg.setdefault((u, v), []).append(float(w[0]))

            new_src, new_dst, new_w = [], [], []
            for (u, v), vals in agg.items():
                if self.edge_attr_reduce == 'sum':
                    val = sum(vals)
                else:  # 'mean'
                    val = sum(vals) / max(1, len(vals))
                new_src.append(u)
                new_dst.append(v)
                new_w.append(val)

            edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=x.device)
            edge_attr = torch.tensor(new_w, dtype=torch.float32, device=x.device).reshape(-1, 1)
            etype = None  # dopo aggregazione non ha più senso mantenere edge_type per arco fuso

        # Costruisci Data omogeneo
        out = Data(x=x, edge_index=edge_index)
        if y is not None:
            out.y = y
        if edge_attr is not None:
            out.edge_attr = edge_attr
        if etype is not None:
            out.edge_type = etype

        # Copia eventuali maschere per-nodo se presenti
        for mask_name in ('train_mask', 'val_mask', 'test_mask', 'is_common', 'was_common_before_dup'):
            if hasattr(node_store, mask_name):
                setattr(out, mask_name, getattr(node_store, mask_name))

        # Trasporta anche eventuali identificativi testuali per comodità
        if hasattr(node_store, 'string_id'):
            out.string_id = list(getattr(node_store, 'string_id'))

        return out


