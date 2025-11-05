# ── plug&play: incolla in un file utils_debug_graph.py o simile ─────────────────
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def _edges_from_edge_index(edge_index: torch.Tensor) -> List[Tuple[int,int]]:
    # edge_index: [2, E] (senza self-loops espliciti)
    return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

def _nx_from_nodes_edges(num_nodes: int, nodes: torch.Tensor, edge_index: torch.Tensor) -> nx.Graph:
    """Costruisce un grafo NetworkX non diretto sul sottinsieme di nodi `nodes` e archi `edge_index` (relativi ai nodi)."""
    # Reindicizza 0..|nodes|-1 per un disegno più compatto
    mapping = {int(g): i for i, g in enumerate(nodes.tolist())}
    E = []
    for u, v in _edges_from_edge_index(edge_index):
        if u in mapping and v in mapping:
            E.append((mapping[u], mapping[v]))
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(E)
    return G

def _plot_components(G: nx.Graph, title: str, seed: int = 42):
    """Plot di un grafo colorando ogni componente connessa con un colore diverso."""
    # layout riproducibile
    pos = nx.spring_layout(G, seed=seed)
    comps = list(nx.connected_components(G))
    # palette semplice (ripetuta se servono più colori)
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    plt.figure(figsize=(7, 6))
    for i, comp in enumerate(comps):
        color = base_colors[i % len(base_colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=list(comp), node_size=180, node_color=color, alpha=0.9, linewidths=0.5, edgecolors="#222222")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # etichette opzionali: commenta se troppo affollate
    # nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def debug_test_partition_visual(
    data: Data,
    parts: Dict[str, object],
    ego_k: int = 1,
    seed: int = 42,
):
    """
    Visualizza la struttura della **partizione di test supervised** in due modalità:
    1) Inductive "plain": grafo di test indotto su soli nodi supervised di test (no aux, no train/val).
    2) Ego-graph batch: unione disgiunta degli ego-graph (r=ego_k) per ciascun nodo di test,
       espansi **solo lungo archi intra-test supervised** (nessun collegamento a nodi train/val o aux).

    NOTA: La modalità 2 replica la semantica dell'inferenza "ego_graph=True" implementata
          come *batch block-diagonale* (gli ego-graph non vengono fusi fra loro).

    Parametri
    ---------
    data  : torch_geometric.data.Data
        Contiene almeno x, y, edge_index, num_nodes.
    parts : dict
        Dizionario dallo splitter con: 'test_mask', 'supervised_mask', 'aux_mask' (opzionale).
    ego_k : int
        Raggio dell’ego-graph (numero di hop).
    seed  : int
        Seed per il layout.

    Output
    ------
    - Stampa: contatori (nodi, archi, #componenti) per entrambe le viste.
    - Due figure matplotlib, una per vista, con componenti colorate diversamente.
    """
    device = data.x.device
    N = data.num_nodes

    # --- nodo di test supervised ---
    keep_test_sup = (parts["test_mask"] & parts["supervised_mask"]).to(torch.bool)
    test_nodes = torch.nonzero(keep_test_sup, as_tuple=False).view(-1).cpu()
    if test_nodes.numel() == 0:
        print("[WARN] Nessun nodo di test supervised trovato.")
        return

    # --- archi intra-test supervised (escludiamo train/val e aux) ---
    ei = data.edge_index.cpu()
    edge_keep_test = keep_test_sup[ei[0]] & keep_test_sup[ei[1]]
    ei_test = ei[:, edge_keep_test]

    # ========== (1) INDUCTIVE "PLAIN": subgrafo indotto su nodi di test ==========
    G_plain = _nx_from_nodes_edges(N, test_nodes, ei_test)
    comps_plain = list(nx.connected_components(G_plain))
    comp_sizes_plain = sorted([len(c) for c in comps_plain], reverse=True)
    print("── Inductive PLAIN (test-only induced) ─────────────────────────")
    print(f"nodes: {G_plain.number_of_nodes()} | edges: {G_plain.number_of_edges()} | components: {len(comps_plain)}")
    print(f"top component sizes: {comp_sizes_plain[:10]}")
    _plot_components(G_plain, title="Inductive TEST (plain): componenti su nodi supervised di test", seed=seed)

    # ========== (2) EGO-GRAPH BATCH: block-diagonal dei singoli egonet ==========
    # costruiamo, per ciascun nodo c, l’ego-graph a r=ego_k limitato al solo ei_test
    # poi prendiamo la disjoint union per visualizzare l'effetto “batch separato”
    graphs = []
    for c in test_nodes.tolist():
        center = torch.tensor([c], dtype=torch.long)
        subset, ei_local, mapping, _ = k_hop_subgraph(
            center, ego_k, ei_test, relabel_nodes=True, num_nodes=N
        )
        # grafico locale
        G_loc = _nx_from_nodes_edges(N, subset.cpu(), ei_local.cpu())
        # assicuriamoci che il centro esista (può capitare sia isolato → grafo 1 nodo)
        graphs.append(G_loc)

    # disjoint union di tutti gli egonet (block-diagonal)
    if graphs:
        G_ego = nx.disjoint_union_all(graphs)
        comps_ego = list(nx.connected_components(G_ego))
        comp_sizes_ego = sorted([len(c) for c in comps_ego], reverse=True)
        print("── Ego-GRAPH BATCH (r = {}, intra-test only) ────────────────".format(ego_k))
        print(f"nodes: {G_ego.number_of_nodes()} | edges: {G_ego.number_of_edges()} | components: {len(comps_ego)}")
        print(f"top component sizes: {comp_sizes_ego[:10]}")
        _plot_components(G_ego, title=f"Ego-graph TEST batch (r={ego_k}): componenti per nodo", seed=seed)
    else:
        print("[WARN] Nessun ego-graph costruito (lista vuota).")


if __name__ == '__main__':

    from data_pipeline import *
    import torch_geometric.transforms as T

    from training_tools.splitter import StratifiedSplitterWithTestHoldout
    from training_tools.task import TaskBinaryNodeClassification
    from training_tools.trainer import Trainer
    from training_tools.validation import CrossValidator
    from training_tools.evaluation import Evaluator


    pre_transform = T.Compose([
        MarkCommonNodes(edge_types=('AD','PD')),
        DuplicateCommonNodesAndRelabel(suffix_ad='_AD', suffix_pd='_PD'),
        ReindexConsecutive(),
        MergeRelationsToHomogeneous(merge=['AD','PD'], edge_attr_reduce='sum'),
        FilterSmallConnectedComponents(topk=2)
    ])

    ds = NeuroDegAnc2VecDataset(root='data', pre_transform=pre_transform)
    data = ds[0]

    splitter = StratifiedSplitterWithTestHoldout(test_size=0.1, n_splits = 5, seed=42, mode='inductive')
    parts = splitter.split(data)

    debug_test_partition_visual(data, parts, ego_k=1)