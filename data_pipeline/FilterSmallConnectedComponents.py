from __future__ import annotations
from typing import List, Optional, Any
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class FilterSmallConnectedComponents(BaseTransform):
    """
    FilterSmallConnectedComponents(topk: Optional[int] = 2,
                                  min_size: Optional[int] = None,
                                  undirected: bool = True)

    Filtra le **componenti connesse minori** da un grafo **omogeneo** `Data`,
    mantenendo solo:
      - le `topk` componenti più grandi, **oppure**
      - le componenti con cardinalità ≥ `min_size`
    (se entrambi specificati, si applicano **entrambi** i vincoli).

    ----------
    Sostituisce nel tuo codice:
      - Tutto il blocco:
        ```
        l = [component for component in nx.connected_components(G)]
        l = sorted(l, key=len, reverse=True)
        for c in l[2:]:  # escludi dalla 3a in poi
            ...
        # Filtra nodi/archi e reset id
        ```
      cioè l’esclusione delle componenti minori (di solito per ripulire il grafo
      dai “frammenti” fuori dal corpo principale AD/PD).

    Cosa aggiunge / miglioramenti:
      - Implementazione in **puro PyTorch** (niente dipendenza da networkx).
      - Parametri chiari (`topk`, `min_size`) e comportamento riproducibile.
      - Reindicizzazione **consecutiva** automatica di nodi e allineamento coerente
        di `edge_index` ed `edge_attr` (se presente). Copia/filtra maschere per-nodo.
    ----------
    Parametri
    - topk: Optional[int]
        Se impostato, mantiene solo le `topk` componenti più grandi (default: 2).
    - min_size: Optional[int]
        Se impostato, mantiene solo componenti di dimensione ≥ `min_size`.
    - undirected: bool
        Se True, calcola le componenti sul grafo **non orientato** (consigliato per CC).
        L’eventuale filtraggio non modifica la direzionalità di `edge_index`.
        (Se il grafo è orientato e vuoi CC fortemente connesse, serve una logica diversa.)

    Output
    - Ritorna un `Data` filtrato con:
        x, y e maschere per-nodo ridotte ai nodi tenuti
        edge_index/edge_attr filtrati e rimappati
        (eventuali `string_id` filtrati coerentemente)

    Note
    - Calcolo CC: implementato con una BFS su liste di adiacenza (robusto per grafi piccoli/medi).
      Per grafi enormi, valuta l’uso di utilità dedicate o versioni CUDA.
    """

    def __init__(self, topk: Optional[int] = 2, min_size: Optional[int] = None, undirected: bool = True):
        self.topk = topk
        self.min_size = min_size
        self.undirected = undirected

    def forward(self, data: Data) -> Data:
        assert isinstance(data, Data), "FilterSmallConnectedComponents atteso su Data omogeneo."
        N = data.num_nodes
        if N == 0 or data.edge_index.numel() == 0:
            return data

        ei = data.edge_index
        if self.undirected:
            # assicura simmetria per il calcolo delle CC
            ei = torch.cat([ei, ei.flip(0)], dim=1)

        # Costruisci adjacency list (CPU, semplice e robusto)
        src = ei[0].cpu().tolist()
        dst = ei[1].cpu().tolist()
        adj: List[List[int]] = [[] for _ in range(N)]
        for u, v in zip(src, dst):
            adj[u].append(v)

        # BFS per componenti connesse
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

        # Ordina per dimensione decrescente
        components.sort(key=len, reverse=True)

        # Seleziona componenti da tenere secondo i vincoli
        keep_sets: List[set] = []
        for idx, comp in enumerate(components):
            if self.topk is not None and idx >= self.topk:
                continue
            if self.min_size is not None and len(comp) < self.min_size:
                continue
            keep_sets.append(set(comp))

        if len(keep_sets) == 0:
            # fallback: tieni la più grande
            keep_nodes = set(components[0])
        else:
            # unione dei set tenuti
            keep_nodes = set().union(*keep_sets)

        # Costruisci mask nodi da tenere
        keep_mask = torch.zeros(N, dtype=torch.bool)
        if len(keep_nodes) > 0:
            keep_mask[list(keep_nodes)] = True
        if torch.all(keep_mask):
            return data  # nulla da filtrare

        # Rimappa old->new consecutivi
        old_to_new = torch.full((N,), -1, dtype=torch.long)
        kept_idx = torch.arange(N, dtype=torch.long)[keep_mask]
        old_to_new[kept_idx] = torch.arange(kept_idx.numel(), dtype=torch.long)

        # Filtra attributi per-nodo (tensor)
        def is_node_tensor(attr: Any) -> bool:
            return isinstance(attr, Tensor) and attr.size(0) == N

        for key, val in list(data.items()):
            if is_node_tensor(val):
                data[key] = val[keep_mask]

        # Filtra eventuali liste/identificativi testuali
        if hasattr(data, 'string_id') and isinstance(data.string_id, list):
            ids = data.string_id
            data.string_id = [ids[i] for i in kept_idx.tolist()]

        # Aggiorna edge_index / edge_attr
        src_e = old_to_new[data.edge_index[0]]
        dst_e = old_to_new[data.edge_index[1]]
        valid = (src_e >= 0) & (dst_e >= 0)
        data.edge_index = torch.stack([src_e[valid], dst_e[valid]], dim=0)
        if hasattr(data, 'edge_attr') and isinstance(data.edge_attr, Tensor):
            data.edge_attr = data.edge_attr[valid]

        # Maschere per-nodo: già filtrate sopra (train/val/test_mask, is_common, ecc.)
        return data
