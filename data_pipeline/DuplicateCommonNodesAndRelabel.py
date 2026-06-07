from __future__ import annotations
from typing import Iterable, List, Optional, Dict, Any
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class DuplicateCommonNodesAndRelabel(BaseTransform):
    """
    DuplicateCommonNodesAndRelabel(suffix_ad='_AD', suffix_pd='_PD',
                                  node_type='protein', edge_types=('AD','PD'),
                                  label_attr='y', id_attr='string_id',
                                  drop_original_common=True)

    Duplica i nodi "condivisi" (marcati da `is_common=True` o con `y==2` se `is_common` non presente),
    creando due copie per nodo:
      - copia destinata alla relazione AD, con `label=0` e `id` suffissato `'_AD'`
      - copia destinata alla relazione PD, con `label=1` e `id` suffissato `'_PD'`

    Ricollega gli archi:
      - per la relazione AD, ogni endpoint che era un nodo comune viene sostituito con l'indice della copia `_AD`
      - per la relazione PD, analogamente con la copia `_PD`

    Opzionalmente marca i nodi originali condivisi come da rimuovere (`drop_mask=True`);
    la loro rimozione e la reindicizzazione consecutiva vengono effettuate dal transform
    successivo `ReindexConsecutive`.

    ----------
    Sostituisce nel tuo codice:
      - «Add "_AD" or "_PD" to shared nodes based on their source» (nodi + archi)
      - «Create the _AD version ... label=0» / «Create the _PD version ... label=1»
      - Parte della logica di riassegnazione coerente tra nodi/archi (la reindicizzazione
        finale è delegata a `ReindexConsecutive`).
    Cosa aggiunge / miglioramenti:
      - Lavora direttamente su `HeteroData`, senza DataFrame intermedi.
      - Copia in blocco *tutti* gli attributi per-nodo con shape [N, ...] (es. `x`, `y`, maschere),
        mantenendo coerenza; per gli attributi-lista (es. `string_id`) gestisce l'estensione.
      - Non usa id negativi o mapping manuali; produce indici nuovi in coda (append)
        e prepara una `drop_mask` per i nodi originali condivisi, per la rimozione nel passo 3.
    ----------
    Parametri
    - suffix_ad / suffix_pd: str
        Suffissi da aggiungere agli identificativi testuali dei nodi duplicati.
    - node_type: str
        Tipo di nodo nel `HeteroData` (default: 'protein').
    - edge_types: Iterable[str]
        Nomi delle relazioni (es. ('AD','PD')).
    - label_attr: str
        Nome dell'attributo etichetta per-nodo (default: 'y').
    - id_attr: Optional[str]
        Nome dell'attributo identificativo testuale (default: 'string_id'); se non presente, i suffissi non vengono applicati.
    - drop_original_common: bool
        Se True, imposta `data[node_type].drop_mask=True` sui nodi comuni originali (verranno rimossi da ReindexConsecutive).
    - detect_common_from_label2: bool
        Se True e `is_common` non è presente, considera `label==2` come "comune".
    """

    def __init__(
        self,
        suffix_ad: str = '_AD',
        suffix_pd: str = '_PD',
        node_type: str = 'protein',
        edge_types: Iterable[str] = ('AD', 'PD'),
        label_attr: str = 'y',
        id_attr: Optional[str] = 'string_id',
        drop_original_common: bool = True,
        detect_common_from_label2: bool = True,
    ):
        self.suffix_ad = suffix_ad
        self.suffix_pd = suffix_pd
        self.node_type = node_type
        self.edge_types = tuple(edge_types)
        self.label_attr = label_attr
        self.id_attr = id_attr
        self.drop_original_common = drop_original_common
        self.detect_common_from_label2 = detect_common_from_label2

    def forward(self, data: HeteroData) -> HeteroData:
        store = data[self.node_type]
        num_nodes: int = store.num_nodes

        # 1) individua nodi comuni (preferisci mask esplicita)
        if hasattr(store, 'is_common'):
            common_mask: Tensor = store.is_common
        elif self.detect_common_from_label2 and hasattr(store, self.label_attr):
            common_mask = (getattr(store, self.label_attr) == 2)
        else:
            raise ValueError("Nessun indicatore di nodi 'comuni': serve `is_common` o label==2.")

        common_idx = torch.nonzero(common_mask, as_tuple=False).view(-1)
        k = int(common_idx.numel())
        if k == 0:
            # niente da duplicare
            if self.drop_original_common:
                # nessun nodo con label 2 dovrebbe rimanere
                if hasattr(store, self.label_attr):
                    y = getattr(store, self.label_attr)
                    assert not torch.any(y == 2), "Nessun nodo comune trovato, ma presenti label==2."
            return data

        # 2) prepara nuovi indici
        new_ad_idx = torch.arange(num_nodes, num_nodes + k, dtype=torch.long)
        new_pd_idx = torch.arange(num_nodes + k, num_nodes + 2 * k, dtype=torch.long)
        new_total = num_nodes + 2 * k

        # 3) duplica tutti gli attributi per-nodo con prima dimensione N
        #    (x, y, maschere, ecc.). Per gli altri tipi (liste) gestisci separatamente.
        def is_node_tensor(attr: Any) -> bool:
            return isinstance(attr, Tensor) and attr.size(0) == num_nodes

        # raccogli le chiavi tensoriali da estendere
        node_keys = [k for k, v in store.items() if is_node_tensor(v)]
        for key in node_keys:
            t: Tensor = store[key]
            dup_ad = t.index_select(0, common_idx)
            dup_pd = t.index_select(0, common_idx)
            # Per label, sovrascrivi le copie secondo (0 per AD, 1 per PD)
            if key == self.label_attr:
                # cast a long
                dup_ad = dup_ad.clone()
                dup_pd = dup_pd.clone()
                dup_ad[:] = 0
                dup_pd[:] = 1
                # l'originale resta (potrà essere droppato dopo)
            store[key] = torch.cat([t, dup_ad, dup_pd], dim=0)

        # Aggiorna is_common: le copie NON sono comuni
        if hasattr(store, 'is_common'):
            is_common_new = torch.cat([
                store.is_common,  # originale
                torch.zeros(k, dtype=torch.bool),  # _AD
                torch.zeros(k, dtype=torch.bool),  # _PD
            ], dim=0)
            store.is_common = is_common_new

        # Attributo testuale: string_id (lista Python tipicamente)
        if self.id_attr is not None and hasattr(store, self.id_attr):
            ids: List[str] = list(getattr(store, self.id_attr))
            # build mapping old_id -> suffixed ids
            ids_ad = []
            ids_pd = []
            for idx in common_idx.tolist():
                base = ids[idx]
                ids_ad.append(f"{base}{self.suffix_ad}")
                ids_pd.append(f"{base}{self.suffix_pd}")
            ids_extended = ids + ids_ad + ids_pd
            setattr(store, self.id_attr, ids_extended)

        # 4) ricollega gli archi: AD verso copie _AD, PD verso copie _PD
        #    Costruisci mappa rapida old_common_idx -> new_*_idx
        map_ad: Dict[int, int] = {int(o): int(n) for o, n in zip(common_idx.tolist(), new_ad_idx.tolist())}
        map_pd: Dict[int, int] = {int(o): int(n) for o, n in zip(common_idx.tolist(), new_pd_idx.tolist())}

        def remap_edge_index(ei: Tensor, mapping: Dict[int, int]) -> Tensor:
            # Sostituzione vettoriale via torch: creiamo una tabella di mapping densa solo dove serve.
            # Approccio: convertiamo in cpu e usiamo numpy per semplicità/velocità sui dizionari.
            device = ei.device
            src = ei[0].cpu().numpy()
            dst = ei[1].cpu().numpy()
            # vectorized replace
            # (nota: operiamo solo su indici comuni per evitare loop Python costosi)
            if mapping:
                # crea array di lookup solo sui comuni
                # trick: usa dict.get per sostituzione condizionale
                src = [mapping.get(int(v), int(v)) for v in src]
                dst = [mapping.get(int(v), int(v)) for v in dst]
                src = torch.tensor(src, dtype=torch.long)
                dst = torch.tensor(dst, dtype=torch.long)
                return torch.stack([src, dst], dim=0).to(device)
            return ei

        # AD edges: mappa sugli indici _AD
        key_ad = (self.node_type, self.edge_types[0], self.node_type)
        data[key_ad].edge_index = remap_edge_index(data[key_ad].edge_index, map_ad)
        # PD edges: mappa sugli indici _PD
        key_pd = (self.node_type, self.edge_types[1], self.node_type)
        data[key_pd].edge_index = remap_edge_index(data[key_pd].edge_index, map_pd)

        # 5) aggiorna num_nodes
        store.num_nodes = new_total

        # 6) marca i nodi comuni originali per la rimozione nel passo successivo
        if self.drop_original_common:
            drop_mask = torch.zeros(new_total, dtype=torch.bool)
            drop_mask[common_idx] = True  # solo gli originali
            # Le copie non si droppano
            setattr(store, 'drop_mask', drop_mask)

        # 7) traccia: i duplicati provengono da nodi comuni
        was_common_before_dup = torch.zeros(new_total, dtype=torch.bool)
        # marchiamo come True solo le copie (utile per analisi a posteriori)
        was_common_before_dup[new_ad_idx] = True
        was_common_before_dup[new_pd_idx] = True
        setattr(store, 'was_common_before_dup', was_common_before_dup)

        return data