from __future__ import annotations
from typing import Iterable, List
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class MarkCommonNodes(BaseTransform):
    """
    MarkCommonNodes(edge_types=('AD','PD'), node_type='protein')

    Identifica i nodi "condivisi" (presenti come endpoint in **entrambe** le relazioni
    specificate) e aggiunge a `data[node_type].is_common` una maschera booleana per-nodo.

    ----------
    Sostituisce nel tuo codice:
      - «Find shared nodes (edge endpoints)»
        ```
        nodes_AD = set(AD_edges['num_id_1']).union(set(AD_edges['num_id_2']))
        nodes_PD = set(PD_edges['num_id_1']).union(set(PD_edges['num_id_2']))
        common_nodes = nodes_AD.intersection(nodes_PD)
        ```
    Cosa aggiunge / miglioramenti:
      - Evita passaggi via pandas/set, lavora direttamente su `edge_index` in torch,
        con `torch.unique` sui nodi toccati da ciascuna relazione.
      - Deposita il risultato come `BoolTensor` (`is_common`) nel node store,
        così i passi successivi non devono ricalcolarlo e possono usarlo in modo
        differenziato.
    ----------
    Parametri
    - edge_types: tuple[str,str] oppure Iterable[str]
        Nomi delle relazioni da considerare come tipi di edge del formato HeteroData.
        Devono esistere come chiavi: (node_type, rel, node_type).
        Default: ('AD','PD').
    - node_type: str
        Tipo di nodo all'interno del tuo HeteroData (default: 'protein').

    Effetti su `data`
    - Aggiunge `data[node_type].is_common: BoolTensor [num_nodes]`.

    Requisiti
    - `data[(node_type, rel, node_type)].edge_index` deve esistere per ogni `rel` passato.
    """

    def __init__(self, edge_types: Iterable[str] = ('AD', 'PD'), node_type: str = 'protein'):
        self.edge_types = tuple(edge_types)
        self.node_type = node_type

    def forward(self, data: HeteroData) -> HeteroData:
        num_nodes = data[self.node_type].num_nodes
        # Insiemi di nodi toccati per ogni relazione
        touched_sets: List[Tensor] = []
        for rel in self.edge_types:
            key = (self.node_type, rel, self.node_type)
            ei: Tensor = data[key].edge_index
            touched = torch.unique(ei.view(-1))
            touched_sets.append(touched)

        # Intersezione: nodi presenti in tutte le relazioni
        common_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if len(touched_sets) > 0:
            common = touched_sets[0]
            for t in touched_sets[1:]:
                # intersect1d su torch:
                common = torch.tensor(sorted(set(common.tolist()).intersection(set(t.tolist()))), dtype=torch.long)
            common_mask[common] = True

        data[self.node_type].is_common = common_mask
        return data