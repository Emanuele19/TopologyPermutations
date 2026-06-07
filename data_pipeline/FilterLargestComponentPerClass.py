from __future__ import annotations
from typing import List, Iterable, Any, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class FilterLargestComponentPerClass(BaseTransform):
    """
    FilterLargestComponentPerClass(class_labels: Iterable[int] = (0, 1),
                                   undirected: bool = True)

    Filtra le componenti connesse di un grafo omogeneo `Data`, mantenendo per ogni
    classe specificata la componente connessa che "rappresenta" meglio quella classe
    (ovvero quella che contiene il maggior numero di nodi con tale etichetta).

    È particolarmente utile in scenari con più patologie per estrarre i cluster 
    principali di ciascuna malattia ed eliminare il rumore strutturale.

    ----------
    Parametri
    - class_labels: Iterable[int]
        Le etichette delle classi per le quali cercare la componente principale.
        Default: (0, 1).
    - undirected: bool
        Se True, calcola le componenti sul grafo non orientato (consigliato).

    Output
    - Ritorna un `Data` filtrato contenente l'unione delle componenti principali
      trovate per ciascuna classe, con indici e attributi riallineati.
    """

    def __init__(self, class_labels: Iterable[int] = (0, 1), undirected: bool = True):
        self.class_labels = list(class_labels)
        self.undirected = undirected

    def forward(self, data: Data) -> Data:
        assert isinstance(data, Data), "FilterLargestComponentPerClass atteso su Data omogeneo."
        N = data.num_nodes
        if N == 0 or data.edge_index.numel() == 0:
            return data

        ei = data.edge_index
        if self.undirected:
            # Simmetrizza per il calcolo delle CC
            ei = torch.cat([ei, ei.flip(0)], dim=1)

        # 1) Costruisci adjacency list (CPU)
        src = ei[0].cpu().tolist()
        dst = ei[1].cpu().tolist()
        adj: List[List[int]] = [[] for _ in range(N)]
        for u, v in zip(src, dst):
            adj[u].append(v)

        # 2) BFS per trovare tutte le componenti connesse
        seen = [False] * N
        components: List[List[int]] = []
        from collections import deque

        for s in range(N):
            if seen[s]:
                continue
            q = deque([s])
            seen[s] = True
            comp = [s]
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
                        comp.append(v)
            components.append(comp)

        # 3) Selezione: per ogni classe, trova la CC con il max numero di nodi di quella classe
        keep_nodes = set()
        y_cpu = data.y.cpu()

        for label in self.class_labels:
            best_comp = []
            max_count = -1
            for comp in components:
                # Conta nodi con la label corrente in questa componente
                count = sum(1 for node_idx in comp if y_cpu[node_idx].item() == label)
                if count > max_count:
                    max_count = count
                    best_comp = comp
            if best_comp:
                keep_nodes.update(best_comp)

        # 4) Seleziona i nodi e applica il filtraggio (Logica di re-indexing coerente)
        keep_mask = torch.zeros(N, dtype=torch.bool)
        if len(keep_nodes) > 0:
            keep_mask[list(keep_nodes)] = True
        else:
            return data # Nessuna classe trovata, restituisco originale o potresti lanciare errore

        # Rimappa old->new consecutivi
        old_to_new = torch.full((N,), -1, dtype=torch.long)
        kept_idx = torch.arange(N, dtype=torch.long)[keep_mask]
        old_to_new[kept_idx] = torch.arange(kept_idx.numel(), dtype=torch.long)

        # Filtra attributi per-nodo (Tensor e Liste)
        for key, val in list(data.items()):
            if isinstance(val, Tensor) and val.size(0) == N:
                data[key] = val[keep_mask]
        if hasattr(data, 'string_id') and isinstance(data.string_id, list):
            data.string_id = [data.string_id[i] for i in kept_idx.tolist()]

        # Filtra e rimappa archi
        src_e, dst_e = old_to_new[data.edge_index[0]], old_to_new[data.edge_index[1]]
        valid = (src_e >= 0) & (dst_e >= 0)
        data.edge_index = torch.stack([src_e[valid], dst_e[valid]], dim=0)
        if hasattr(data, 'edge_attr') and isinstance(data.edge_attr, Tensor):
            data.edge_attr = data.edge_attr[valid]

        return data