import torch_geometric.transforms as T
from data_pipeline import *
from pathlib import Path
import json
import pandas as pd
import torch

# 1) PRE-TRANSFORM (una volta → cache)
pre_t = T.Compose([
    MarkCommonNodes(edge_types=('AD','PD')),
    DuplicateCommonNodesAndRelabel(suffix_ad='_AD', suffix_pd='_PD'),
    ReindexConsecutive(),
    MergeRelationsToHomogeneous(merge=['AD','PD'], edge_attr_reduce='sum'),
    FilterSmallConnectedComponents(topk=2)
])


def tensors_to_python_list(t: torch.Tensor):
    # Converte su CPU -> list nativo (evita problemi di serializzazione)
    return t.detach().cpu().tolist()

def export_graph_to_csv(g, out_dir: str = "pre_processing_output",
                        nodes_csv: str = "nodes_merged.csv",
                        edges_csv: str = "edges_merged.csv"):
    """
    Esporta:
      - nodes_merged.csv con colonne: id (int), string_id (str, se presente), embedding (json array)
      - edges_merged.csv con colonne: src (int), dst (int), edge_attr (float/array, se presente), edge_type (int, se presente)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ===== NODI =====
    num_nodes = g.num_nodes
    node_ids = list(range(num_nodes))

    if hasattr(g, "string_id"):
        string_ids = list(getattr(g, "string_id"))
        assert len(string_ids) == num_nodes, "string_id length mismatch with num_nodes"
    else:
        string_ids = [f"node_{i}" for i in node_ids]

    # embedding come JSON (una cella con l'array)
    x = g.x
    assert x.dim() == 2, "g.x atteso 2D [N, D]"
    # converti riga per riga per non esplodere in colonne
    embeddings_json = [json.dumps(tensors_to_python_list(row)) for row in x]

    nodes_df = pd.DataFrame({
        "id": node_ids,
        "STRING_id": string_ids,
        "embedding": embeddings_json,
    })

    # ===== ARCHI =====
    ei = g.edge_index
    src = tensors_to_python_list(ei[0])
    dst = tensors_to_python_list(ei[1])

    data = {
        "src": src,
        "dst": dst,
    }

    # edge_attr: se presente e ha 1 col -> float; se >1 col -> json array
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        ea = g.edge_attr.detach().cpu()
        if ea.dim() == 1 or (ea.dim() == 2 and ea.size(1) == 1):
            edge_attr_col = ea.view(-1).tolist()
        else:
            edge_attr_col = [json.dumps(row.tolist()) for row in ea]
        data["edge_attr"] = edge_attr_col

    # edge_type: se presente
    if hasattr(g, "edge_type"):
        data["edge_type"] = tensors_to_python_list(g.edge_type)

    edges_df = pd.DataFrame(data)

    # ===== SALVA =====
    nodes_path = out / nodes_csv
    edges_path = out / edges_csv
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"[OK] Salvati:\n - {nodes_path}\n - {edges_path}")



dataset = NeuroDegAnc2VecDataset(root="data", pre_transform=pre_t)
g = dataset[0]  # Data con x,y,edge_index, train/val/test_mask
export_graph_to_csv(g, out_dir='.')

new_csv = pd.read_csv('./nodes_merged.csv')
old_csv = pd.read_csv('./pre_processing_output/duplicated_nodes_main_components.csv')

new_ds = new_csv[['STRING_id', 'embedding']].set_index('STRING_id')['embedding'].to_dict()
old_ds = old_csv[['STRING_id', 'GO_embeddings']].set_index('STRING_id')['GO_embeddings'].to_dict()

for k in new_ds.keys():
    if new_ds[k] != old_ds[k]:
        print(f"Valore con chiave {k} di tipi {type(new_ds[k])} e {type(old_ds[k])} non corrispondono")

