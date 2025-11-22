from __future__ import annotations
import re
from typing import Optional, Dict, List, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
import networkx as nx
import random


def _make_simple_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Rende il grafo semplice e non orientato: rimuove self-loop, deduplica, simmetrizza."""
    ei, _ = remove_self_loops(edge_index)
    # tieni solo una direzione per deduplicare (i<j)
    i, j = ei
    mask_upper = i < j
    iu = torch.where(mask_upper, i, j)
    ju = torch.where(mask_upper, j, i)
    ei_u = torch.stack([iu, ju], dim=0)
    # rimuovi duplicati
    ei_u = torch.unique(ei_u, dim=1)
    # simmetrizza
    ei_ud = to_undirected(ei_u, num_nodes=num_nodes)
    return ei_ud


def _nx_from_edge_index(edge_index: Tensor, num_nodes: int) -> nx.Graph:
    """Crea un grafo semplice di NetworkX mantenendo gli indici 0..N-1 come label nodi."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # usa solo metà superiore per evitare duplicati
    i, j = edge_index
    mask = i < j
    edges = torch.stack([i[mask], j[mask]], dim=0).T.tolist()
    G.add_edges_from(edges)
    return G


def _edge_index_from_nx(G: nx.Graph, num_nodes: int, device: torch.device) -> Tensor:
    """Converte nx.Graph -> edge_index non orientato, semplice."""
    if G.number_of_edges() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    import numpy as np
    e = np.array(G.edges(), dtype=int)
    ei = torch.as_tensor(e.T, dtype=torch.long, device=device)
    ei = _make_simple_undirected(ei, num_nodes=num_nodes)
    return ei


def _double_edge_swap_safe(
    G: nx.Graph, nswap: int, max_tries: int, ensure_connected: bool, seed: int | random.Random
) -> nx.Graph:
    """Esegue il double-edge-swap con fallback progressivo se non converge."""
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)
    H = G.copy()
    try:
        if ensure_connected:
            nx.connected_double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rng)
        else:
            nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rng)
        return H
    except nx.NetworkXError:
        # fallback: riduci nswap a metà finché riesce
        cur = max(nswap // 2, 1)
        while cur >= 1:
            try:
                if ensure_connected:
                    nx.connected_double_edge_swap(H, nswap=cur, max_tries=max(max_tries, 10_000), seed=rng)
                else:
                    nx.double_edge_swap(H, nswap=cur, max_tries=max(max_tries, 10_000), seed=rng)
                return H
            except nx.NetworkXError:
                cur //= 2
        return G.copy()  # nessun rewiring possibile: ritorna l’originale


def rewire_degree_preserving_complete(
    data: Data,
    *,
    seed: int = 0,
    swaps_per_edge: float = 10.0,
    max_tries_factor: float = 10.0,
    ensure_connected: bool = False,
    stratify_by_suffix: bool = True,
    suffix_regex: str = r"_(AD|PD)$",
) -> Data:
    """
    Rewiring degree-preserving su tutto il grafo (non stratificato per label).

    Parametri
    ---------
    data : Data
        Oggetto PyG con campi almeno: x, edge_index, y.
        (Opz.) string_id: List[str] con suffisso '_AD'/'_PD' per la stratificazione.
    seed : int
        Seed per la randomizzazione.
    swaps_per_edge : float
        Numero target di swap ~ swaps_per_edge * |E_half| (dove E_half è il numero di archi non duplicati).
    max_tries_factor : float
        Fattore moltiplicativo per max_tries del double-edge-swap.
    ensure_connected : bool
        Se True, usa connected_double_edge_swap (più restrittivo).
    stratify_by_suffix : bool
        Se True, fa il rewiring separatamente per i nodi con suffisso '_AD' e '_PD' in string_id.
        Gli archi cross AD↔PD vengono lasciati invariati (conservativi).
    suffix_regex : str
        Regex per riconoscere i suffissi di malattia nei nomi dei nodi.

    Ritorna
    -------
    Data
        Una *copia* di `data` con `edge_index` rewired (edge_attr ricreato a 1).

    Note
    ----
    - Il grafo risultante è semplice, non orientato, senza self-loop.
    - edge_attr viene riassegnato a 1 (Tensor [E,1]) per neutralità.
    """
    g = data.clone()
    device = g.x.device if isinstance(g.x, torch.Tensor) else torch.device("cpu")
    N = g.num_nodes
    # normalizza edge_index a grafo semplice non orientato
    ei = _make_simple_undirected(g.edge_index.to('cpu'), num_nodes=N)

    # partizione per suffisso (se attiva)
    if stratify_by_suffix and hasattr(g, "string_id"):
        names: List[str] = list(getattr(g, "string_id"))
        suf_re = re.compile(suffix_regex)
        is_AD = torch.tensor([bool(suf_re.search(n) and n.endswith("_AD")) for n in names], dtype=torch.bool)
        is_PD = torch.tensor([bool(suf_re.search(n) and n.endswith("_PD")) for n in names], dtype=torch.bool)
        # nodi che non matchano il suffisso vanno in "others"
        is_other = ~(is_AD | is_PD)
    else:
        is_AD = torch.zeros(N, dtype=torch.bool)
        is_PD = torch.zeros(N, dtype=torch.bool)
        is_other = torch.ones(N, dtype=torch.bool)

    # Costruisci nx global e separa insiemi
    G_all = _nx_from_edge_index(ei, num_nodes=N)

    # crea subgrafi per le tre “intra” e lascia i cross fermi (conservativo)
    parts: Dict[str, Dict] = {}
    for tag, m in {"AD": is_AD, "PD": is_PD, "OT": is_other}.items():
        nodes = torch.nonzero(m, as_tuple=False).view(-1).tolist()
        if len(nodes) == 0:
            continue
        H = G_all.subgraph(nodes).copy()
        parts[tag] = {"nodes": nodes, "G": H}

    # rewiring su ciascun subgrafo “intra”
    rng = random.Random(seed)
    for tag, d in parts.items():
        H = d["G"]
        E = H.number_of_edges()
        if E == 0:
            continue
        nswaps = max(1, int(swaps_per_edge * E))
        max_tries = max(10_000, int(max_tries_factor * nswaps))
        parts[tag]["G_rew"] = _double_edge_swap_safe(H, nswap=nswaps, max_tries=max_tries,
                                                     ensure_connected=ensure_connected, seed=rng)
    # ricompone il grafo: sostituisce solo le intra-edges, lascia invariati i cross
    G_new = nx.Graph()
    G_new.add_nodes_from(range(N))

    # aggiungi intra-edges rewired
    for tag, d in parts.items():
        if "G_rew" in d:
            G_new.add_edges_from(d["G_rew"].edges())

    # aggiungi i cross edges invariati
    # (cross = tutti gli edges originali non intra-AD, non intra-PD, non intra-OT)
    intra_all = set()
    for tag, d in parts.items():
        intra_all.update((min(u, v), max(u, v)) for (u, v) in d["G"].edges())
    cross_edges = []
    for (u, v) in G_all.edges():
        key = (min(u, v), max(u, v))
        if key not in intra_all:
            cross_edges.append((u, v))
    G_new.add_edges_from(cross_edges)

    # back to edge_index
    ei_new = _edge_index_from_nx(G_new, num_nodes=N, device=device)
    g.edge_index = ei_new

    # edge_attr neutro = 1 (shape [E,1])
    E_new = g.edge_index.size(1) // 2  # metà superiore *2 per simmetrizzazione
    g.edge_attr = torch.ones((2 * E_new, 1), dtype=torch.float32, device=device)

    return g


# ---------------------- Esempio d'uso ----------------------
# g_rewired = rewire_degree_preserving(
#     data,
#     seed=42,
#     swaps_per_edge=10.0,
#     ensure_connected=False,
#     stratify_by_suffix=True,     # usa suffissi _AD/_PD in string_id
#     suffix_regex=r"_(AD|PD)$",
# )
# # Ora puoi addestrare la GNN su g_rewired (transductive) e confrontare con il grafo reale.

from typing import List, Tuple
import random
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
import networkx as nx


def _edge_index_to_half_undirected(edge_index: Tensor) -> Tensor:
    """
    Porta edge_index a rappresentazione non orientata "half":
    - rimuove self-loop
    - normalizza ogni arco come (min(u,v), max(u,v))
    - rimuove duplicati
    Restituisce edge_index [2, E_half] con u<v per ogni colonna.
    """
    ei, _ = remove_self_loops(edge_index)
    i, j = ei
    u = torch.minimum(i, j)
    v = torch.maximum(i, j)
    ei_half = torch.stack([u, v], dim=0)
    ei_half = torch.unique(ei_half, dim=1)
    return ei_half


def rewire_degree_preserving_stratified(
    data: Data,
    *,
    seed: int = 0,
    swaps_per_edge: float = 10.0,
    max_tries_factor: float = 10.0,
) -> Data:
    """
    Rewiring degree-preserving separato per classe 0 e 1 usando double-edge-swap.

    Assunzioni:
    - data.y contiene solo label 0 e 1.
    - NON esistono archi cross-classe (0-1); se ci sono → assert.

    Procedura:
    - Estrae sottografo 0<->0 e 1<->1 (non orientati, semplici).
    - Su ciascuno esegue nx.double_edge_swap con nswap ~= swaps_per_edge * E.
    - Ricompone il grafo globale con gli archi rewired (sempre intra-classe).

    Proprietà:
    - Numero di archi per classe preservato esattamente.
    - Degree sequence per classe preservata esattamente (salvo errori NX).
    - Nessun arco cross-classe (restano zero).
    """
    g = data.clone()
    device = g.x.device if isinstance(g.x, torch.Tensor) else torch.device("cpu")
    N = g.num_nodes

    # controlla label
    y = g.y.to('cpu')
    y_unique = set(y.unique().tolist())
    assert y_unique.issubset({0, 1}), f"Le label devono essere solo 0/1, trovate {y_unique}"

    # edge_index half non orientato
    ei_half = _edge_index_to_half_undirected(g.edge_index.to('cpu'))
    i, j = ei_half

    mask0 = (y == 0)
    mask1 = (y == 1)

    # partiziona archi half in intra-0, intra-1, cross (che non dovrebbero esistere)
    intra0_edges: List[Tuple[int, int]] = []
    intra1_edges: List[Tuple[int, int]] = []
    cross_edges: List[Tuple[int, int]] = []

    for u, v in zip(i.tolist(), j.tolist()):
        if mask0[u] and mask0[v]:
            intra0_edges.append((u, v))  # u<v già
        elif mask1[u] and mask1[v]:
            intra1_edges.append((u, v))
        else:
            cross_edges.append((u, v))

    assert len(cross_edges) == 0, (
        f"Trovati {len(cross_edges)} archi non intra-classe (0-1 o altro). "
        "Il setting 'no cross' non è rispettato."
    )

    rng = random.Random(seed)

    # helper per rewiring con double_edge_swap
    def _rewire_block(nodes: List[int], edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(edges) == 0:
            return []

        H = nx.Graph()
        H.add_nodes_from(nodes)
        H.add_edges_from(edges)
        E = H.number_of_edges()
        nswap = max(1, int(swaps_per_edge * E))
        max_tries = max(10_000, int(max_tries_factor * nswap))

        try:
            nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rng)
        except nx.NetworkXError:
            # fallback: riduci nswap gradualmente
            cur = nswap // 2
            while cur >= 1:
                try:
                    nx.double_edge_swap(H, nswap=cur, max_tries=max_tries, seed=rng)
                    break
                except nx.NetworkXError:
                    cur //= 2
        # ritorna lista di archi half (u<v)
        new_edges = []
        for u, v in H.edges():
            e = (u, v) if u < v else (v, u)
            new_edges.append(e)
        return new_edges

    idx0 = torch.nonzero(mask0, as_tuple=False).view(-1).tolist()
    idx1 = torch.nonzero(mask1, as_tuple=False).view(-1).tolist()

    new_intra0 = _rewire_block(idx0, intra0_edges)
    new_intra1 = _rewire_block(idx1, intra1_edges)

    # ricomponi tutti gli archi half nel grafo intero
    all_edges_half = set()
    for e in new_intra0:
        all_edges_half.add(e)
    for e in new_intra1:
        all_edges_half.add(e)

    if len(all_edges_half) == 0:
        ei_half_new = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edges_arr = np.array(list(all_edges_half), dtype=np.int64)
        ei_half_new = torch.as_tensor(edges_arr.T, dtype=torch.long, device=device)

    # per PyG puoi tenere il grafo come half (un solo arco per coppia)
    # oppure simmetrizzarlo; scegli in base a come era il tuo data originale
    # Se il tuo 'data.edge_index' originale aveva 660 archi e li intendi "undirected",
    # probabilmente ogni edge è già singolo → NON simmetrizzare.
    g.edge_index = ei_half_new.to(device)

    # edge_attr neutro: 1 per ogni edge
    g.edge_attr = torch.ones((g.edge_index.size(1), 1), dtype=torch.float32, device=device)

    return g


def _rewire_block(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    *,
    swaps_per_edge: float,
    max_tries_factor: float,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Versione ROBUSTA di rewiring degree-preserving su un sottografo (nodes, edges),
    basata su nx.double_edge_swap con fallback.

    - Parte da un grafo semplice H con i nodi `nodes` e archi `edges`.
    - Tenta di effettuare ~swaps_per_edge * E swap (dove E = #edges del blocco).
    - Se NetworkX non riesce a trovare abbastanza swap (NetworkXError / NetworkXAlgorithmError),
      riduce progressivamente nswap (nswap -> nswap//2 -> ...) finché >=1.
    - Se non riesce nemmeno con pochi swap, ritorna semplicemente `edges` (nessun rewiring).

    Ritorna:
    - nuova lista di archi half (u<v) se rewiring ha successo,
    - altrimenti gli archi originali.
    """
    if len(edges) == 0:
        return []

    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    E_block = H.number_of_edges()
    if E_block == 0:
        return []

    nswap_target = max(1, int(swaps_per_edge * E_block))

    # tentiamo con nswap_target, poi nswap_target//2, poi //4, ... finché >=1
    cur_nswap = nswap_target
    success = False

    while cur_nswap >= 1 and not success:
        max_tries = max(10_000, int(max_tries_factor * cur_nswap))
        try:
            nx.double_edge_swap(H, nswap=cur_nswap, max_tries=max_tries, seed=rng)
            success = True
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            # riduci il numero di swap richiesti e riprova
            cur_nswap //= 2

    if not success:
        # impossibile fare anche pochi swap: ritorna edges originali
        print("[WARN] impossibile fare swap")
        return edges.copy()

    # ritorna lista di archi half (u<v)
    new_edges = []
    for u, v in H.edges():
        e = (u, v) if u < v else (v, u)
        new_edges.append(e)
    return new_edges


def rewire_degree_preserving_stratified_connected(
    data: Data,
    *,
    seed: int = 0,
    swaps_per_edge: float = 10.0,
    max_tries_factor: float = 10.0,
) -> Data:
    """
    Rewiring degree-preserving STRATIFICATO per classi 0 e 1 su grafo connesso.

    Assunzioni / comportamento:
    - data.y contiene label in {0,1,2}, dove:
        0 = classe AD-only (esempio)
        1 = classe PD-only
        2 = nodi comuni / aux
      (eventuali altre label → AssertionError).

    - Il grafo può avere archi cross-classe:
        0-1, 0-2, 1-2.
      Questi archi NON vengono rewired: restano invariati.

    - Il rewiring viene eseguito SOLO sui blocchi intra-classe:
        * 0<->0  (blocco di label 0)
        * 1<->1  (blocco di label 1)
      Il blocco 2<->2 (label 2) resta esattamente come nell'originale.

    Procedura:
    - Converte edge_index in forma non orientata "half" (u<v, no self-loop, no duplicati).
    - Partiziona gli archi in:
        intra0 : 0<->0
        intra1 : 1<->1
        intra2 : 2<->2   (NON modificato)
        cross  : tutto il resto (0-1, 0-2, 1-2) (NON modificato)
    - Per intra0 e intra1:
        * costruisce un sottografo NetworkX,
        * esegue `_rewire_block` (versione robusta).
    - Ricompone edge_index con:
        * intra0 rewired,
        * intra1 rewired,
        * intra2 originali,
        * cross originali.

    Proprietà:
    - Per blocchi 0 e 1, degree sequence preservata (nei limiti di double_edge_swap).
    - Blocchi 2<->2 e archi cross-classe immutati.
    - edge_index risultante in forma "half" (un edge per coppia non orientata).
    """
    g = data.clone()
    device = g.x.device if isinstance(g.x, torch.Tensor) else torch.device("cpu")

    # controlla label
    y = g.y.to('cpu')
    y_unique = set(y.unique().tolist())
    assert y_unique.issubset({0, 1, 2}), f"Le label devono essere solo 0/1/2, trovate {y_unique}"

    # edge_index non orientato "half" (u<v, no self-loop, no duplicati)
    ei_half = _edge_index_to_half_undirected(g.edge_index.to('cpu'))
    i, j = ei_half

    mask0 = (y == 0)
    mask1 = (y == 1)
    mask2 = (y == 2)

    # partiziona archi half
    intra0_edges: List[Tuple[int, int]] = []
    intra1_edges: List[Tuple[int, int]] = []
    intra2_edges: List[Tuple[int, int]] = []
    cross_edges: List[Tuple[int, int]] = []

    for u, v in zip(i.tolist(), j.tolist()):
        if mask0[u] and mask0[v]:
            intra0_edges.append((u, v))  # u < v
        elif mask1[u] and mask1[v]:
            intra1_edges.append((u, v))
        elif mask2[u] and mask2[v]:
            intra2_edges.append((u, v))  # QUESTI NON LI TOCCHIAMO
        else:
            cross_edges.append((u, v))   # archi misti (0-1, 0-2, 1-2), invariati

    rng = random.Random(seed)

    # nodi per blocchi 0 e 1
    idx0 = torch.nonzero(mask0, as_tuple=False).view(-1).tolist()
    idx1 = torch.nonzero(mask1, as_tuple=False).view(-1).tolist()
    # i nodi di label 2 non servono a _rewire_block perché intra2_edges resta com'è

    # rewiring SOLO per blocchi 0 e 1 (label 2 resta intatto)
    new_intra0 = _rewire_block(
        idx0,
        intra0_edges,
        swaps_per_edge=swaps_per_edge,
        max_tries_factor=max_tries_factor,
        rng=rng,
    )
    new_intra1 = _rewire_block(
        idx1,
        intra1_edges,
        swaps_per_edge=swaps_per_edge,
        max_tries_factor=max_tries_factor,
        rng=rng,
    )
    # blocco 2: nessun rewiring
    new_intra2 = intra2_edges

    # ricompone tutti gli archi half nel grafo intero
    all_edges_half = set()
    for e in new_intra0:
        all_edges_half.add(e)
    for e in new_intra1:
        all_edges_half.add(e)
    for e in new_intra2:
        all_edges_half.add(e)
    for e in cross_edges:
        all_edges_half.add(e)

    if len(all_edges_half) == 0:
        ei_half_new = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edges_arr = np.array(list(all_edges_half), dtype=np.int64)
        ei_half_new = torch.as_tensor(edges_arr.T, dtype=torch.long, device=device)

    # edge_index finale: manteniamo la convenzione "half" (un edge per coppia non orientata)
    g.edge_index = ei_half_new.to(device)

    # edge_attr neutro: 1 per ogni edge
    g.edge_attr = torch.ones((g.edge_index.size(1), 1), dtype=torch.float32, device=device)

    return g



# ---------------------------------------------------


from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


def _sample_simple_undirected_edges_on_nodes(
    nodes: List[int],
    num_edges: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Campiona num_edges archi non orientati semplici (u<v) su un insieme di nodi dato.
    - nessun self-loop
    - nessun multi-edge
    - grafo non orientato (un solo edge per coppia)

    Se num_edges > num_possible, solleva AssertionError.
    """
    n = len(nodes)
    if num_edges == 0 or n < 2:
        return []

    # lista di tutte le possibili coppie (u<v) sui nodi forniti
    # complessità O(n^2), ok per PPI piccoli
    possible_edges: List[Tuple[int, int]] = []
    for idx_i in range(n):
        u = nodes[idx_i]
        for idx_j in range(idx_i + 1, n):
            v = nodes[idx_j]
            possible_edges.append((u, v))

    if num_edges > len(possible_edges):
        raise AssertionError(
            f"Richiesti {num_edges} archi ma solo {len(possible_edges)} possibili coppie su {n} nodi."
        )

    # scegli num_edges coppie a caso senza rimpiazzo
    idx = rng.choice(len(possible_edges), size=num_edges, replace=False)
    sampled = [possible_edges[k] for k in idx]
    return sampled


def rewire_non_degree_preserving_stratified(
    data: Data,
    *,
    seed: int = 0,
) -> Data:
    """
    Null model NON degree-preserving separato per classe 0 e 1.

    Assunzioni:
    - data.y contiene SOLO etichette 0 e 1.
    - NON esistono archi cross-classe (0-1); se ci sono -> assert.
    - Il grafo è memorizzato in forma 'half' (un solo edge per coppia non orientata).
      (Nel tuo caso E_real(undirected)=660 con edge_index.size(1)=660, quindi è vero.)

    Procedura:
    - Estrae gli archi intra-0 (0<->0) e intra-1 (1<->1) dal grafo reale.
    - Conta il numero di archi per ciascuna classe: E0, E1.
    - Per ogni classe:
        * prende i nodi di quella classe,
        * campiona E_c archi random (u<v) da tutte le coppie possibili,
          SENZA preservare i gradi per nodo.
    - Ricompone il grafo globale come unione degli archi intra-0 e intra-1 randomizzati.
      Nessun arco cross-classe.

    Proprietà:
    - Stesso numero di archi per classe (E0, E1) del grafo reale.
    - Degree sequence NON preservata (null model più "forte").
    - Nessun arco cross 0-1 (come nel grafo reale).

    Ritorna:
    - Copia di `data` con edge_index rewired (half, u<v) e edge_attr neutro (=1).
    """
    g = data.clone()
    device = g.x.device if isinstance(g.x, torch.Tensor) else torch.device("cpu")
    N = g.num_nodes

    # controlla label
    y = g.y.to('cpu')
    #y_unique = set(y.unique().tolist())
    #assert y_unique.issubset({0, 1}), f"Le label devono essere solo 0/1, trovate {y_unique}"

    # edge_index non orientato half (u<v)
    ei_half = _edge_index_to_half_undirected(g.edge_index.to('cpu'))
    i, j = ei_half

    mask0 = (y == 0)
    mask1 = (y == 1)

    # partiziona archi in intra-0, intra-1, cross (che NON dovrebbero esistere)
    intra0_edges: List[Tuple[int, int]] = []
    intra1_edges: List[Tuple[int, int]] = []
    cross_edges: List[Tuple[int, int]] = []

    for u, v in zip(i.tolist(), j.tolist()):
        if mask0[u] and mask0[v]:
            intra0_edges.append((u, v))  # u<v
        elif mask1[u] and mask1[v]:
            intra1_edges.append((u, v))
        else:
            cross_edges.append((u, v))

    assert len(cross_edges) == 0, (
        f"Trovati {len(cross_edges)} archi non intra-classe (0-1 o altro). "
        "Il setting 'no cross' non è rispettato."
    )

    E0 = len(intra0_edges)
    E1 = len(intra1_edges)

    # nodi per classe
    idx0 = torch.nonzero(mask0, as_tuple=False).view(-1).tolist()
    idx1 = torch.nonzero(mask1, as_tuple=False).view(-1).tolist()

    rng = np.random.default_rng(seed)

    # campiona nuovi archi intra-classe SENZA preservare i gradi
    new_intra0 = _sample_simple_undirected_edges_on_nodes(idx0, E0, rng)
    new_intra1 = _sample_simple_undirected_edges_on_nodes(idx1, E1, rng)

    # unisci tutto (sempre half, u<v)
    all_edges_half = set()
    for e in new_intra0:
        all_edges_half.add(e)
    for e in new_intra1:
        all_edges_half.add(e)

    if len(all_edges_half) == 0:
        ei_half_new = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edges_arr = np.array(list(all_edges_half), dtype=np.int64)
        ei_half_new = torch.as_tensor(edges_arr.T, dtype=torch.long, device=device)

    # edge_index finale: manteniamo la convenzione 'half' (un edge per coppia)
    g.edge_index = ei_half_new.to(device)

    # edge_attr neutro: 1 per ogni edge
    g.edge_attr = torch.ones((g.edge_index.size(1), 1), dtype=torch.float32, device=device)

    return g


def rewire_non_degree_preserving_stratified_complete(
    data: Data,
    *,
    seed: int = 0,
) -> Data:
    """
    Null model NON degree-preserving stratificato per classi 0 e 1,
    lasciando invariati i nodi/componente di label 2 e TUTTI gli archi cross-classe.

    Assunzioni:
    - data.y contiene label in {0,1,2}, dove:
        0 = classe AD-only
        1 = classe PD-only
        2 = nodi comuni / aux
      (altre label -> AssertionError).
    - Il grafo è memorizzato in forma 'half' (un solo edge per coppia non orientata),
      oppure comunque viene portato in quella forma da _edge_index_to_half_undirected.

    Procedura:
    - Estrae gli archi:
        intra-0 : 0<->0
        intra-1 : 1<->1
        other   : tutti gli altri (0-1, 0-2, 1-2, 2-2)
    - Conta E0 = |intra-0|, E1 = |intra-1|.
    - Per ogni classe c ∈ {0,1}:
        * prende i nodi con label c,
        * campiona E_c archi random (u<v) da tutte le coppie possibili di nodi di classe c,
          SENZA preservare i gradi per nodo (Erdős–Rényi vincolato).
    - Ricompone il grafo globale come unione di:
        * archi intra-0 randomizzati,
        * archi intra-1 randomizzati,
        * tutti gli archi in 'other' invariati (compresi 2<->2 e cross).

    Proprietà:
    - Stesso numero di archi intra-0 e intra-1 dell'originale.
    - Degree sequence NON preservata per nodi di classe 0 e 1.
    - Nodi di classe 2 e archi che li coinvolgono restano identici all'originale.
    """
    g = data.clone()
    device = g.x.device if isinstance(g.x, torch.Tensor) else torch.device("cpu")

    # controlla label
    y = g.y.to('cpu')
    y_unique = set(y.unique().tolist())
    assert y_unique.issubset({0, 1, 2}), f"Le label devono essere solo 0/1/2, trovate {y_unique}"

    # edge_index non orientato half (u<v)
    ei_half = _edge_index_to_half_undirected(g.edge_index.to('cpu'))
    i, j = ei_half

    mask0 = (y == 0)
    mask1 = (y == 1)
    mask2 = (y == 2)

    # partiziona archi:
    # - intra0 : 0<->0
    # - intra1 : 1<->1
    # - other  : tutto il resto (0-1, 0-2, 1-2, 2-2) -> INVARIATI
    intra0_edges: List[Tuple[int, int]] = []
    intra1_edges: List[Tuple[int, int]] = []
    other_edges: List[Tuple[int, int]] = []

    for u, v in zip(i.tolist(), j.tolist()):
        if mask0[u] and mask0[v]:
            intra0_edges.append((u, v))  # u<v
        elif mask1[u] and mask1[v]:
            intra1_edges.append((u, v))
        else:
            other_edges.append((u, v))

    E0 = len(intra0_edges)
    E1 = len(intra1_edges)

    # nodi per classe (label 2 NON viene rewired)
    idx0 = torch.nonzero(mask0, as_tuple=False).view(-1).tolist()
    idx1 = torch.nonzero(mask1, as_tuple=False).view(-1).tolist()

    rng = np.random.default_rng(seed)

    # campiona nuovi archi intra-classe SENZA preservare i gradi
    new_intra0 = _sample_simple_undirected_edges_on_nodes(idx0, E0, rng) if E0 > 0 else []
    new_intra1 = _sample_simple_undirected_edges_on_nodes(idx1, E1, rng) if E1 > 0 else []

    # unisci tutto (sempre half, u<v)
    all_edges_half = set()
    for e in new_intra0:
        all_edges_half.add(e)
    for e in new_intra1:
        all_edges_half.add(e)
    for e in other_edges:     # label 2 e cross-classe invariati
        all_edges_half.add(e)

    if len(all_edges_half) == 0:
        ei_half_new = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edges_arr = np.array(list(all_edges_half), dtype=np.int64)
        ei_half_new = torch.as_tensor(edges_arr.T, dtype=torch.long, device=device)

    # edge_index finale: manteniamo la convenzione 'half' (un edge per coppia)
    g.edge_index = ei_half_new.to(device)

    # edge_attr neutro: 1 per ogni edge
    g.edge_attr = torch.ones((g.edge_index.size(1), 1), dtype=torch.float32, device=device)

    return g
