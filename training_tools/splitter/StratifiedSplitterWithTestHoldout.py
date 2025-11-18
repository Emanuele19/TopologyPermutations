from __future__ import annotations
from typing import Dict, List, Literal
import re
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from collections import deque

from training_tools.splitter.HoldoutMode import HoldoutMode


class StratifiedSplitterWithTestHoldout:
    """
    Splitter senza 'grouping' (equivalente al caso in cui ogni gruppo ha cardinalità 1):
    - Esclude i nodi ausiliari (_AD/_PD) dal test.
    - Stratifica il test sugli unici nodi supervised.
    - Esegue K-fold stratificato sugli supervised rimasti (train pool).
    - (Opz.) Fornisce edge mask per training/validation induttivo.

    Convenzioni:
    - supervised_mask = nodi NON duplicati (no _AD/_PD), con y ∈ {0,1}
    - aux_mask        = nodi duplicati (_AD/_PD) → mai in test, sempre nel grafo, fuori da loss/metriche

    Esempio di utilizzo (transduttivo):
    >>> splitter = StratifiedSplitterWithTestHoldout(test_size=0.1, n_splits=5, seed=42,  mode='transductive')
    >>> parts = splitter.split(g)

    >>> for k in range(splitter.n_splits):
    ...     train_mask = (parts["cv_fold"] != k) & parts["train_pool_mask"] & parts["supervised_mask"]
    ...     val_mask   = (parts["cv_fold"] == k) & parts["train_pool_mask"] & parts["supervised_mask"]

    Esempio di utilizzo (induttivo):
    >>> splitter = StratifiedSplitterWithTestHoldout(test_size=0.1, n_splits=5, seed=42,  mode='inductive')
    >>> parts = splitter.split(g)

    >>> for k in range(splitter.n_splits):
    ...    train_mask = (parts["cv_fold"] != k) & parts["train_pool_mask"] & parts["supervised_mask"]
    ...    val_mask   = (parts["cv_fold"] == k) & parts["train_pool_mask"] & parts["supervised_mask"]
    ...    ei_train = g.edge_index[:, parts["edge_keep_mask_train"][k]]
    ...    logits   = model(g.x, ei_train)  # train step
    ...     ...
    ...    ei_val   = g.edge_index[:, parts["edge_keep_mask_val"][k]]
    ...    logits_v = model(g.x, ei_val)    # val step

    """

    def __init__(self, test_size: float = 0.1, n_splits: int = 5, seed: int = 42,
                 name_attr: str = "string_id", suffix_regex: str = r"_(AD|PD)$",
                 mode: Literal['inductive', 'transductive'] = 'inductive',
                 aux_in_test: bool = True):
        assert 0 < test_size < 1 and n_splits >= 2
        self.test_size = float(test_size)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.name_attr = name_attr
        self.suffix_re = re.compile(suffix_regex)
        self.mode = mode

    def split(self, data: Data, holdout_mode: HoldoutMode = HoldoutMode.RANDOM, aux_in_test: bool = True) -> Dict[str, object]:
        """
        Esegue lo split nested (test hold-out + K-fold interno) per classificazione binaria
        a livello nodo su un grafo singolo, rispettando la policy:
        - i nodi duplicati ausiliari (es. con suffisso "_AD"/"_PD") **non** entrano mai in test,
            ma restano nel grafo per fornire contesto topologico (message passing);
        - lo split (test e K-fold) è eseguito **solo** sui nodi supervised (non duplicati).

        Parametri
        ----------
        data : torch_geometric.data.Data
            Grafo omogeneo con almeno:
            - `x`: Tensor [N, F], feature per nodo
            - `y`: LongTensor [N], etichette per nodo (binario: {0,1} per i supervised)
            - `edge_index`: LongTensor [2, E], archi
            - (opz.) `string_id`: List[str] di lunghezza N; i duplicati terminano con "_AD"/"_PD"
            - (opz.) `was_common_before_dup`: BoolTensor [N], True per nodi duplicati derivati da nodi comuni

        Ritorna
        -------
        dict
            Dizionario con i seguenti campi (tutti indicizzati su N nodi, salvo dove diversamente specificato):

            - **"supervised_mask"** : BoolTensor [N]
                Maschera True sui nodi *supervised* (non duplicati) usati per la supervisione.
                Definizione: `~aux_mask`. I nodi supervised devono avere `y ∈ {0,1}`.
                *Uso tipico:* limita loss e metriche a questo sottoinsieme insieme alle mask di train/val/test.

            - **"aux_mask"** : BoolTensor [N]
                Maschera True sui nodi *ausiliari* (duplicati, es. con suffisso "_AD"/"_PD" oppure
                `was_common_before_dup=True`). Questi nodi **non** entrano mai nel test né nella loss,
                ma restano **presenti nel grafo** durante train/val per contribuire al message passing.
                *Invarianti:* `supervised_mask = ~aux_mask`.

            - **"test_mask"** : BoolTensor [N]
                Maschera True sui nodi supervised selezionati per il **test hold-out** stratificato.
                Per costruzione `test_mask & aux_mask = False`. Viene usata **solo** al termine del K-fold
                per la valutazione finale (nessun tuning su questo insieme).

            - **"train_pool_mask"** : BoolTensor [N]
                Maschera True sui nodi presenti nel *pool di training/validation*. Definizione:
                `(supervised_mask & ~test_mask) | aux_mask`. In altre parole, include:
                * tutti i supervised **non** in test (candidati a train/val),
                * **tutti** gli ausiliari (sempre presenti come contesto).
                *Uso tipico:* vincola lo spazio su cui costruire train/val in ciascun fold.

            - **"cv_fold"** : LongTensor [N]
                Per i soli nodi supervised nel `train_pool_mask`, contiene l'indice di fold `k` assegnato
                da `0` a `n_splits-1` (stratificato su `y`). Per gli altri nodi (test o aux) vale `-1`.
                *Derivazione tipica delle mask per un fold `k`:*
                - `train_mask = (cv_fold != k) & train_pool_mask & supervised_mask`
                - `val_mask   = (cv_fold == k) & train_pool_mask & supervised_mask`

            - **"edge_keep_mask_train"** : List[BoolTensor [E]] (presente solo se ` mode='inductive'`)
                Lista di lunghezza `n_splits`; per ciascun fold `k`, una maschera booleana sugli archi da
                **mantenere durante il training** del fold `k`. Un arco (u,v) è True se **entrambi** gli endpoint
                appartengono a `nodes_train_phase = train_mask_supervised_k ∪ aux_mask`, dove
                `train_mask_supervised_k = (cv_fold != k) & train_pool_supervised`.
                *Uso tipico (induttivo):* `ei_train = edge_index[:, edge_keep_mask_train[k]]`.

            - **"edge_keep_mask_val"** : List[BoolTensor [E]] (presente solo se ` mode='inductive'`)
                Lista di lunghezza `n_splits`; per ciascun fold `k`, maschera booleana sugli archi da
                **mantenere durante la validazione** del fold `k`. Un arco è True se entrambi gli endpoint
                appartengono a `nodes_val_phase = val_mask_supervised_k ∪ aux_mask`, dove
                `val_mask_supervised_k = (cv_fold == k) & train_pool_supervised`.
                *Uso tipico (induttivo):* `ei_val = edge_index[:, edge_keep_mask_val[k]]`.

        Note
        ----
        - Gli ausiliari **non** compaiono mai in `test_mask` né in `train_mask`/`val_mask`; restano nel grafo come
        contesto strutturale (message passing) ma sono esclusi da loss e metriche.
        - La stratificazione è binaria su `y` dei supervised; in implementazione di riferimento si usa
        `sklearn.model_selection.StratifiedKFold` per il K-fold sui supervised del train pool.
        """
        N = data.num_nodes
        y: Tensor = data.y.to(torch.long)

        # 1) aux vs supervised (via suffisso nel nome)
        names = list(getattr(data, self.name_attr)) if hasattr(data, self.name_attr) else [f"n{i}" for i in range(N)]
        aux_mask = torch.tensor([bool(self.suffix_re.search(n)) for n in names], dtype=torch.bool)
        supervised_mask = (~aux_mask)

        # safety: supervised devono essere binari
        y_sup_unique = y[supervised_mask].unique().tolist()
        assert set(y_sup_unique).issubset({0, 1}), f"Label supervised non binarie: {y_sup_unique}"

        rng = np.random.default_rng(self.seed)

        # 2) Test hold-out STRATIFICATO sui supervised
        sup_idx = torch.nonzero(supervised_mask, as_tuple=False).view(-1).cpu().numpy()
        sup_y = y[supervised_mask].cpu().numpy().astype(int)

        aux_idx = torch.nonzero(aux_mask, as_tuple=False).view(-1).cpu().numpy() if aux_in_test else None

        # stratified split semplice per nodi
        if holdout_mode == HoldoutMode.RANDOM:
            test_sel = self._stratified_holdout_indices(sup_idx, sup_y, frac=self.test_size, rng=rng)
        elif holdout_mode == HoldoutMode.BFS:
            test_sel = self._bfs_stratified_holdout_indices(
                indices_sup=sup_idx,
                indices_aux=aux_idx,
                labels=sup_y,
                edge_index=data.edge_index,
                frac=self.test_size,
                rng=rng,)
            
        test_mask = torch.zeros(N, dtype=torch.bool)
        test_mask[test_sel] = True
        if not aux_in_test:
            test_mask &= supervised_mask

        # 3) Train pool: supervised rimanenti + aux
        train_pool_supervised = supervised_mask & (~test_mask)
        train_pool_mask = train_pool_supervised | aux_mask

        # 4) K-fold STRATIFICATO sui supervised del train pool
        train_sup_idx = torch.nonzero(train_pool_supervised, as_tuple=False).view(-1).cpu().numpy()
        train_sup_y   = y[train_pool_supervised].cpu().numpy().astype(int)

        cv_fold = torch.full((N,), -1, dtype=torch.long)
        fold_assign = self._stratified_kfold_assign(train_sup_idx, train_sup_y, K=self.n_splits, seed=self.seed)
        for idx, f in fold_assign.items():
            cv_fold[idx] = int(f)

        out = dict(
            supervised_mask=supervised_mask,
            aux_mask=aux_mask,
            train_pool_mask=train_pool_mask,
            test_mask=test_mask,
            cv_fold=cv_fold,
        )

        if self.mode == 'transductive':
            return out

        # 5) Induttivo: edge mask per TRAIN/VAL per ogni fold
        ei = data.edge_index
        edge_keep_mask_train: List[Tensor] = []
        edge_keep_mask_val: List[Tensor] = []
        for k in range(self.n_splits):
            train_sup_k = (cv_fold != k) & train_pool_supervised
            val_sup_k   = (cv_fold == k) & train_pool_supervised

            nodes_train_phase = train_sup_k | aux_mask
            nodes_val_phase   = val_sup_k | aux_mask

            keep_train = nodes_train_phase[ei[0]] & nodes_train_phase[ei[1]]
            keep_val   = nodes_val_phase[ei[0]] & nodes_val_phase[ei[1]]

            edge_keep_mask_train.append(keep_train)
            edge_keep_mask_val.append(keep_val)

        out["edge_keep_mask_train"] = edge_keep_mask_train
        out["edge_keep_mask_val"] = edge_keep_mask_val
        return out

    # -------- helpers --------
    @staticmethod
    def _stratified_holdout_indices(indices: np.ndarray, labels: np.ndarray, frac: float, rng) -> np.ndarray:
        """Ritorna una maschera che rappresenta l'holdout di proporzioni frac su indices stratificato per labels"""
        sel = []
        for l in np.unique(labels):
            label_indices = indices[labels == l]
            k = max(1, int(round(frac * len(label_indices))))
            k = min(k, len(label_indices))
            if k > 0:
                sel.append(rng.choice(label_indices, size=k, replace=False))
        return np.unique(np.concatenate(sel)) if sel else np.array([], dtype=indices.dtype)
    
    @staticmethod
    def _bfs_stratified_holdout_indices_old(
        indices: np.ndarray,          # indici GLOBALI dei soli supervised
        labels: np.ndarray,           # label 0/1 allineate a 'indices'
        edge_index: torch.Tensor,     # [2, E] archi globali
        frac: float,                  # frazione di supervised da mettere in test
        rng,                          # np.random.Generator (seed già fissato a monte)
        backfill_random_if_needed: bool = True,
    ) -> np.ndarray:
        """
        Holdout ~frac, con DUE seed (uno per classe).
        - Calcola la quota per classe: k0, k1 (round frac * |classe|).
        - Sceglie un seed casuale in ciascuna classe.
        - Esegue due BFS indipendenti (classe 0 e classe 1) finché non raggiunge le rispettive quote.
        - Se una BFS non copre tutta la quota, opzionalmente completa a caso dentro la stessa classe.

        Ritorna: indici GLOBALI selezionati per il test.
        """
        if len(indices) == 0:
            return np.array([], dtype=int)

        # ---- Quote per classe ----
        classes = np.unique(labels)
        targets = {}
        for c in classes:
            n_c = int((labels == c).sum())
            k_c = max(1, int(round(frac * n_c)))
            k_c = min(k_c, n_c)
            targets[int(c)] = k_c

        # ---- Mappa globale<->locale e adiacenza sui supervised ----
        idx_gl2loc = {int(g): i for i, g in enumerate(indices)} # ridondante se gli id sono ordinati e continui
        u = edge_index[0].cpu().numpy()
        v = edge_index[1].cpu().numpy()
        mask_sup = np.isin(u, indices) & np.isin(v, indices) # solo archi che collegano nodi supervised
        u_sup, v_sup = u[mask_sup], v[mask_sup] # i supervised collegati dagli archi della riga precedente

        adj = [[] for _ in range(len(indices))]
        for a, b in zip(u_sup, v_sup):
            ia, ib = idx_gl2loc[int(a)], idx_gl2loc[int(b)]
            adj[ia].append(ib)
            adj[ib].append(ia)

        # ---- helper: BFS confinata alla classe 'c' ----
        def bfs_(seed_loc: int, k_target: int) -> list[int]:
            """Ritorna posizioni locali selezionate nella classe c (max k_target) via BFS da seed_loc."""
            sel = []
            visited = np.zeros(len(indices), dtype=bool)
            q = deque([seed_loc])
            visited[seed_loc] = True
            while q and len(sel) < k_target:
                cur = q.popleft()
                sel.append(cur)
                neigh = adj[cur]
                if len(neigh) > 1:
                    neigh = neigh.copy()
                    rng.shuffle(neigh) # Per evitare ordine deterministico
                for nb in neigh:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            return sel

        selected_loc_all = []

        # ---- classe 0 ----
        loc0 = np.where(labels == 0)[0]
        target0 = targets.get(0, 0)

        selected0: list[int] = []
        remaining_target0 = target0

        pool0 = set(loc0.tolist())

        # accumula via BFS finché raggiungi il target o non hai più candidati utili
        while remaining_target0 > 0 and pool0:
            # scegli un seed casuale tra i rimanenti (non ancora selezionati)
            seed0_loc = int(rng.choice(list(pool0)))
            sel0_once = bfs_(seed0_loc, k_target=remaining_target0)

            # tieni solo nodi della stessa classe e non ancora scelti
            sel0_once = [i for i in sel0_once if i in pool0]

            if not sel0_once:
                # nessun progresso da questo seed: non riprovare questo seed
                pool0.discard(seed0_loc)
                continue

            # aggiorna selezione cumulativa
            selected0.extend(sel0_once)
            pool0.difference_update(sel0_once)
            remaining_target0 = target0 - len(selected0)

            if backfill_random_if_needed and remaining_target0 > 0 and pool0:
                print(f"[WARN] backfill classe 0: {len(selected0) = }, {target0 = }")
            else:
                break

        # aggiungi al set totale
        selected_loc_all.extend(selected0)

        # se non hai raggiunto il target e vuoi saperlo
        if len(selected0) < target0:
            print(f"[WARN] classe 0: raggiunti {len(selected0)}/{target0} nodi con BFS multi-seed.")

        # ---- classe 1 ----
        loc1 = np.where(labels == 1)[0]
        target1 = targets.get(1, 0)

        selected1: list[int] = []
        remaining_target1 = target1
        pool1 = set(loc1.tolist())

        while remaining_target1 > 0 and pool1:
            seed1_loc = int(rng.choice(list(pool1)))
            sel1_once = bfs_(seed1_loc, k_target=remaining_target1)
            sel1_once = [i for i in sel1_once if i in pool1]

            if not sel1_once:
                pool1.discard(seed1_loc)
                continue

            selected1.extend(sel1_once)
            pool1.difference_update(sel1_once)
            remaining_target1 = target1 - len(selected1)

            if backfill_random_if_needed and remaining_target1 > 0 and pool1:
                print(f"[WARN] backfill classe 1: {len(selected1) = }, {target1 = }")
            else:
                break

        selected_loc_all.extend(selected1)

        if len(selected1) < target1:
            print(f"[WARN] classe 1: raggiunti {len(selected1)}/{target1} nodi con BFS multi-seed.")

        # ---- ritorna indici GLOBALI ordinati/uniques ----
        if not selected_loc_all:
            return np.array([], dtype=int)
        selected_glob = indices[np.unique(np.array(selected_loc_all, dtype=int))]
        return selected_glob
    
    @staticmethod
    def _bfs_stratified_holdout_indices(
        indices_sup: np.ndarray,        # indici GLOBALI dei soli supervised
        labels: np.ndarray,             # label 0/1 allineate a 'indices'
        edge_index: torch.Tensor,       # [2, E] archi globali
        frac: float,                    # frazione di supervised da mettere in test
        rng,                            # np.random.Generator (seed già fissato a monte)
        indices_aux: np.ndarray = None, # indici GLOBALI dei nodi ausiliari (se None, non agisce)
    ) -> np.ndarray:
        """
        Holdout ~frac, con DUE seed (uno per classe).
        - Calcola la quota per classe: k0, k1 (round frac * |classe|).
        - Sceglie un seed casuale in ciascuna classe.
        - Esegue due BFS indipendenti (classe 0 e classe 1) finché non raggiunge le rispettive quote.
        - Se una BFS non copre tutta la quota, opzionalmente completa a caso dentro la stessa classe.

        Ritorna: indici GLOBALI selezionati per il test.
        """
        if len(indices_sup) == 0:
            return np.array([], dtype=int)

        # ---- Quote per classe ----
        classes = np.unique(labels)
        targets = {}
        for c in classes:
            n_c = int((labels == c).sum())
            k_c = max(1, int(round(frac * n_c)))
            k_c = min(k_c, n_c)
            targets[int(c)] = k_c

        # ---- Mappa globale<->locale e adiacenza sui supervised ----
        idx_gl2loc = {int(g): i for i, g in enumerate(indices_sup)} # ridondante se gli id sono ordinati e continui
        u = edge_index[0].cpu().numpy()
        v = edge_index[1].cpu().numpy()
        mask_sup = np.isin(u, indices_sup) & np.isin(v, indices_sup) # solo nodi supervised collegati
        u_sup, v_sup = u[mask_sup], v[mask_sup] # id dei nodi di cui sopra

        adj = [[] for _ in range(len(indices_sup))]
        for a, b in zip(u_sup, v_sup):
            ia, ib = idx_gl2loc[int(a)], idx_gl2loc[int(b)]
            adj[ia].append(ib)
            adj[ib].append(ia)

        idx_gl2loc_aux = dict()
        if isinstance(indices_aux, np.ndarray): # (vuol dire not None)se si vogliono includere anche i nodi ausiliari
            idx_gl2loc_aux = {int(g): int(g) for _, g in enumerate(indices_aux)}
            idx_gl2loc |= idx_gl2loc_aux # ridondante se gli id sono ordinati e continui
            mask_aux = np.isin(u, indices_aux) & np.isin(v, indices_aux)    # Tutti i nodi aux collegati
            mask_aux |= np.isin(u, indices_aux) & np.isin(v, indices_sup)   # Tutti i nodi aux collegati ad un nodo sup
            mask_aux |= np.isin(u, indices_sup) & np.isin(v, indices_aux)   # Inverso di sopra per adirezionalità (ridondante probabilmente)
            u_add, v_add = u[mask_aux], v[mask_aux]

            adj += [[] for _ in range(len(idx_gl2loc))] # aggiungi entry ai vettori di adiacenza
            for a, b in zip(u_add, v_add):
                ia, ib = idx_gl2loc[int(a)], idx_gl2loc[int(b)]
                adj[ia].append(ib)
                adj[ib].append(ia)


        # ---- helper: BFS confinata alla classe 'c' ----
        def bfs_(seed_loc: int, k_target: int) -> list[int]:
            """Ritorna posizioni locali selezionate nella classe c (max k_target) via BFS da seed_loc."""
            sel = []
            count = 0
            visited_len = len(indices_sup) + len(idx_gl2loc_aux)
            visited = np.zeros(visited_len, dtype=bool)
            q = deque([seed_loc])
            visited[seed_loc] = True
            while q and count < k_target:
                cur = q.popleft()
                sel.append(cur) 
                if cur not in idx_gl2loc_aux: # conta solo i nodi supervised
                    count += 1
                neigh = adj[cur]
                if len(neigh) > 1:
                    neigh = neigh.copy()
                    if isinstance(indices_aux, np.ndarray): # Randomizza il vicinato sup e sposta quello aux alla fine (prediligi nodi sup)
                        sup_neigh = [n for n in neigh if n not in idx_gl2loc_aux]
                        aux_neigh = [n for n in neigh if n in idx_gl2loc_aux]
                        rng.shuffle(sup_neigh)
                        neigh = sup_neigh + aux_neigh
                    else:
                        rng.shuffle(neigh) # Per evitare ordine deterministico
                for nb in neigh:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            return sel

        selected_loc_all = []

        # ---- classe 0 ----
        loc0 = np.where(labels == 0)[0]
        target0 = targets.get(0, 0)

        seed0_loc = int(rng.choice(list(loc0)))
        sel0_once = bfs_(seed0_loc, k_target=target0)

        if len([n for n in sel0_once if n not in idx_gl2loc_aux]) < target0:
            print(f"[WARN] c'è bisogno di backfill (classe 0)")
        
        selected_loc_all.extend(sel0_once)

        # ---- classe 1 ----
        loc1 = np.where(labels == 1)[0]
        target1 = targets.get(1, 0)

        seed1_loc = int(rng.choice(list(loc1)))
        sel1_once = bfs_(seed1_loc, k_target=target1)

        if len([n for n in sel1_once if n not in idx_gl2loc_aux]) < target1:
            print(f"[WARN] c'è bisogno di backfill (classe 1)")

        selected_loc_all.extend(sel1_once)

        # ---- ritorna indici GLOBALI ordinati/uniques ----
        if not selected_loc_all:
            return np.array([], dtype=int)
        
        global_indices = np.hstack([indices_sup, indices_aux]) if isinstance(indices_aux, np.ndarray) else indices_sup

        selected_glob = global_indices[np.unique(np.array(selected_loc_all, dtype=int))]
        return selected_glob


    @staticmethod
    def _stratified_kfold_assign(indices: np.ndarray, labels: np.ndarray, K: int, seed: int):
        """
        Assegna a ciascun indice supervised un numero di fold in 0..K-1 usando StratifiedKFold.
        Ritorna: dict {indice: fold_id}
        """
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        assign = {}
        # sklearn si aspetta X,y allineati; qui X sono solo indici “placeholder”
        X = np.zeros((len(indices), 1), dtype=np.int8)
        y = labels
        for fold_id, (train_idx_pos, val_idx_pos) in enumerate(skf.split(X, y)):
            # val_idx_pos sono POSIZIONI nell’array 'indices', non gli indici globali dei nodi
            val_indices = indices[val_idx_pos]
            for idx in val_indices:
                assign[int(idx)] = int(fold_id)
        return assign

