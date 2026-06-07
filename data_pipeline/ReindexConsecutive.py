from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class ReindexConsecutive(BaseTransform):
    """
    ReindexConsecutive(node_type='protein', remove_with_drop_mask=True)

    Esegue una reindicizzazione **consecutiva** degli indici di nodo, aggiornando
    coerentemente *tutte* le relazioni/edge_index e filtrando gli attributi per-nodo.
    Se presente `data[node_type].drop_mask`, rimuove i nodi marcati come True (es. gli
    originali "comuni" dopo la duplicazione).

    ----------
    Sostituisce nel tuo codice:
      - «Reassign numeric ids for duplicated nodes (consistency between nodes and edges)»
        (le mappe `to_change_map` e gli aggiornamenti consequenziali su nodi e archi)
      - Reset degli indici finali coerenti su nodes/edges.
    Cosa aggiunge / miglioramenti:
      - Gestione generica su HeteroData: aggiorna *tutti* gli `edge_index` per ogni relazione.
      - Filtra in blocco gli attributi per-nodo (tensor e liste), garantendo coerenza dimensionale.
      - Nessuna dipendenza da pandas/CSV; opera in memoria su strutture torch.
    ----------
    Parametri
    - node_type: str
        Tipo di nodo da reindicizzare (default: 'protein').
    - remove_with_drop_mask: bool
        Se True e `drop_mask` presente, rimuove i nodi marcati; altrimenti ignora `drop_mask`.

    Effetti su `data`
    - `data[node_type].num_nodes` aggiornato al nuovo numero.
    - Tutti gli attributi per-nodo filtrati/reindicizzati.
    - Ogni `edge_index` delle relazioni `(node_type, rel, node_type)` rimappato ai nuovi id.
    - Rimozione di `drop_mask` (consumata) e aggiornamento eventuale di flag derivati.
    """

    def __init__(self, node_type: str = 'protein', remove_with_drop_mask: bool = True):
        self.node_type = node_type
        self.remove_with_drop_mask = remove_with_drop_mask

    def forward(self, data: HeteroData) -> HeteroData:
        store = data[self.node_type]
        num_nodes = store.num_nodes

        # 1) determina mask dei nodi da tenere
        if self.remove_with_drop_mask and hasattr(store, 'drop_mask'):
            drop_mask: Tensor = store.drop_mask
            assert drop_mask.dtype == torch.bool and drop_mask.numel() == num_nodes
            keep_mask = ~drop_mask
        else:
            keep_mask = torch.ones(num_nodes, dtype=torch.bool)

        # se tutto da tenere e già consecutivo, nulla da fare
        if torch.all(keep_mask):
            # comunque assicuriamo che l'edge_index non referenzi nodi fuori range (non dovrebbe)
            return data

        # 2) costruisci mapping old->new consecutivo
        old_to_new = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
        old_idx = torch.arange(num_nodes, dtype=torch.long)
        kept_idx = old_idx[keep_mask]
        old_to_new[kept_idx] = torch.arange(kept_idx.numel(), dtype=torch.long)

        # 3) filtra tutti gli attributi per-nodo [N, ...]
        def is_node_tensor(attr: Any) -> bool:
            return isinstance(attr, Tensor) and attr.size(0) == num_nodes

        node_keys = [k for k, v in store.items() if is_node_tensor(v)]
        for key in node_keys:
            t: Tensor = store[key]
            store[key] = t[keep_mask]

        # 3b) lista di identificativi testuali (se presente)
        if hasattr(store, 'string_id') and isinstance(store.string_id, list):
            ids = store.string_id
            store.string_id = [ids[i] for i in kept_idx.tolist()]

        # 3c) aggiorna num_nodes
        new_num = int(keep_mask.sum().item())
        store.num_nodes = new_num

        # 4) rimappa tutti gli edge_index delle relazioni pertinenti
        #    (eliminiamo anche archi incidenti a nodi droppati)
        for key in data.edge_types:
            s, rel, d = key
            if s == self.node_type and d == self.node_type:
                ei: Tensor = data[key].edge_index
                src = old_to_new[ei[0]]
                dst = old_to_new[ei[1]]
                # filtra archi invalidi (quelli che toccavano nodi droppati hanno -1)
                valid = (src >= 0) & (dst >= 0)
                data[key].edge_index = torch.stack([src[valid], dst[valid]], dim=0)
                # se sono presenti edge_attr, filtrali coerentemente
                if hasattr(data[key], 'edge_attr') and isinstance(data[key].edge_attr, Tensor):
                    data[key].edge_attr = data[key].edge_attr[valid]

        # 5) pulizia: drop_mask consumata
        if hasattr(store, 'drop_mask'):
            delattr(store, 'drop_mask')

        return data