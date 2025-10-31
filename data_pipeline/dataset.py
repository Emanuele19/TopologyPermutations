from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
import re
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from data_pipeline.anc2vec_utils import load_anc2vec_npz, merge_embeddings

from torch.serialization import add_safe_globals, safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])


class NeuroDegAnc2VecDataset(InMemoryDataset):
    """
    Costruisce un HeteroData con:
      - nodi 'protein' con feature x (anc2vec) e y in {0,1,2}
      - edge types:
          ('protein','AD','protein') con edge_attr 'experimentally_determined_interaction'
          ('protein','PD','protein') con edge_attr analogo

    Assunzioni sui CSV:
      nodes.csv: colonne ["name", "Gene Ontology IDs", ...]
      edges.csv: colonne ["name", "experimentally_determined_interaction"]
        dove 'name' è una stringa "P1 (interacts with) P2"

    NOTE PER IL FUTURO (HeteroData):
    È una struttura annidata composta da:
    - node stores indicizzabili per tipo (es. protein, gene, ...)
    - edge stores indicizzabili per tupla del tipo (src, relation, dst) 
        (es. prot1 AD prot2 vuol dire che la proteina prot1 ha un adge verso prot2 nella relazione Alzheimer)
    
    Attributi utili di HeteroData:
    - node_types
    - edge_types

    La costruzione di questo HeteroData è fatta impostando:
        data["protein"].x = `embedding`
        data["protein"].y = `label`
        data["protein"].string_id = `lista di STRING_id`
        data[('protein', 'relation', 'protein')].edge_index = `src_index, dst_index`
        data[('protein', 'relation', 'protein')].edge_attr  = `experimentally_determined_interaction`
    """
    def __init__(self,
                 root: str,
                 ad_nodes_csv: str = "networks/AD_nodes.csv",
                 pd_nodes_csv: str = "networks/PD_nodes.csv",
                 ad_edges_csv: str = "networks/AD_edges.csv",
                 pd_edges_csv: str = "networks/PD_edges.csv",
                 anc2vec_npz_path: str = "data_pipeline/anc2vec_go_embeddings_v1.npz",
                 transform=None, pre_transform=None):
        self._cfg = {
            "ad_nodes_csv": ad_nodes_csv,
            "pd_nodes_csv": pd_nodes_csv,
            "ad_edges_csv": ad_edges_csv,
            "pd_edges_csv": pd_edges_csv,
            "anc2vec_npz_path": anc2vec_npz_path
        }
        super().__init__(root, transform, pre_transform)
        # --- LOAD con allowlist + weights_only=False ---
        processed_path = self.processed_paths[0]
        try:
            with safe_globals([BaseStorage, NodeStorage, EdgeStorage]):
                self.data, self.slices = torch.load(processed_path, weights_only=False)
        except Exception as e:
            # fallback: prova a ricaricare senza context (alcuni env bastano) o suggerisci reprocess
            self.data, self.slices = torch.load(processed_path, weights_only=False)


    # ----- PyG paths -----
    @property
    def raw_file_names(self):
        # Le path sono relative a root/data/raw/
        return [
            "networks/AD_nodes.csv",
            "networks/PD_nodes.csv",
            "networks/AD_edges.csv",
            "networks/PD_edges.csv",
        ]

    @property
    def processed_file_names(self):
        return ["hetero.pt"]

    def download(self):
        # I file devono già essere in data/raw/
        pass

    # ----- helpers -----
    @staticmethod
    def _extract_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
        # estrae "node1","node2" da 'name' = "X (interacts with) Y"
        pairs = df_edges["name"].str.extract(r'(\S+)\s+\(interacts with\)\s+(\S+)')
        pairs.columns = ["node1", "node2"]
        out = pd.DataFrame({
            "node1": pairs["node1"],
            "node2": pairs["node2"],
            "experimentally_determined_interaction": df_edges["experimentally_determined_interaction"].astype(float)
        })
        return out

    @staticmethod
    def _split_go_ids(series: pd.Series) -> pd.Series:
        # "GO:0008150;GO:0003674" -> lista ["GO:0008150", "GO:0003674"]
        return series.fillna("").map(lambda s: [t.strip() for t in s.split(";") if t.strip() != ""])

    # ----- main process -----
    def process(self):
        """
        Costruisce un HeteroData unico a partire dai CSV raw e dagli embedding anc2vec
        pre-esportati in formato NPZ (Python >=3.9, nessuna dipendenza da anc2vec a runtime).

        Richiede in self._cfg["anc2vec_npz_path"] il percorso all'NPZ, ad es.:
        data/external/anc2vec_go_embeddings_v1.npz
        """
        from data_pipeline.anc2vec_utils import load_anc2vec_npz, merge_embeddings

        # ----- 0) config e input files
        ad_nodes_csv = self._cfg["ad_nodes_csv"]
        pd_nodes_csv = self._cfg["pd_nodes_csv"]
        ad_edges_csv = self._cfg["ad_edges_csv"]
        pd_edges_csv = self._cfg["pd_edges_csv"]
        anc2vec_npz_path = Path(self._cfg['anc2vec_npz_path'])
        if not anc2vec_npz_path.exists():
            raise FileNotFoundError(
                f"File NPZ con embedding anc2vec non trovato: {anc2vec_npz_path}. "
                f"Esporta prima gli embedding (py3.6) e riprova."
            )

        # ----- 1) carica nodi
        ad_nodes = pd.read_csv(ad_nodes_csv).rename(columns={"name": "STRING_id"})
        pd_nodes = pd.read_csv(pd_nodes_csv).rename(columns={"name": "STRING_id"})

        # ----- 2) carica e parsa archi
        ad_edges_raw = pd.read_csv(ad_edges_csv, usecols=["name", "experimentally_determined_interaction"])
        pd_edges_raw = pd.read_csv(pd_edges_csv, usecols=["name", "experimentally_determined_interaction"])
        ad_edges = self._extract_edges(ad_edges_raw)  # -> columns: node1,node2,experimentally_determined_interaction
        pd_edges = self._extract_edges(pd_edges_raw)

        # ----- 3) GO list per nodo
        ad_nodes["GO_list"] = self._split_go_ids(ad_nodes["Gene Ontology IDs"])
        pd_nodes["GO_list"] = self._split_go_ids(pd_nodes["Gene Ontology IDs"])

        # ----- 4) etichette 0/1 e merge; overlap -> 2
        ad_nodes["label"] = 0
        pd_nodes["label"] = 1
        combined = pd.concat(
            [ad_nodes[["STRING_id", "GO_list", "label"]],
             pd_nodes[["STRING_id", "GO_list", "label"]]],
            ignore_index=True
        )

        # label: se presente in entrambi -> 2 ; GO_list: unione dei termini
        lab = combined.groupby("STRING_id")["label"].agg(lambda s: 2 if s.nunique() > 1 else s.iloc[0])
        go_agg = combined.groupby("STRING_id")["GO_list"].agg(
            lambda lists: lists.iloc[0] if len(lists) == 1 else list({g for L in lists for g in L})
        )
        merged = pd.DataFrame(
            {"STRING_id": lab.index, "label": lab.values, "GO_list": go_agg.values}
        ).reset_index(drop=True)

        # ----- 5) carica embedding anc2vec (NPZ) e featurizza con somma
        terms, vectors, go2idx = load_anc2vec_npz(str(anc2vec_npz_path))  # terms [N], vectors [N,D]

        x_mat = np.vstack([
            merge_embeddings(go_list, go2idx, vectors) for go_list in merged["GO_list"]
        ])  # [num_nodes, D]
        x = torch.from_numpy(x_mat).float()
        y = torch.tensor(merged["label"].values, dtype=torch.long)

        # ----- 6) mappa STRING_id -> indice
        id2idx = {sid: i for i, sid in enumerate(merged["STRING_id"].tolist())}

        def map_edges(df_e: pd.DataFrame):
            src = df_e["node1"].map(id2idx)
            dst = df_e["node2"].map(id2idx)
            mask = src.notna() & dst.notna()
            src = src[mask].astype(int).to_numpy()
            dst = dst[mask].astype(int).to_numpy()
            w = df_e.loc[mask, "experimentally_determined_interaction"].to_numpy(dtype=np.float32)
            edge_index = np.stack([src, dst], axis=0)  # [2, E]
            return edge_index, w

        ad_edge_index_np, ad_w = map_edges(ad_edges)
        pd_edge_index_np, pd_w = map_edges(pd_edges)

        # ----- 7) costruisci HeteroData
        data = HeteroData()
        data["protein"].x = x
        data["protein"].y = y
        data["protein"].string_id = merged["STRING_id"].tolist()

        data[("protein", "AD", "protein")].edge_index = torch.from_numpy(ad_edge_index_np).long()
        data[("protein", "AD", "protein")].edge_attr  = torch.from_numpy(ad_w).reshape(-1, 1)

        data[("protein", "PD", "protein")].edge_index = torch.from_numpy(pd_edge_index_np).long()
        data[("protein", "PD", "protein")].edge_attr  = torch.from_numpy(pd_w).reshape(-1, 1)

        # ----- 8) pre_filter / pre_transform
        # Nota: su HeteroData, pre_filter se presente deve accettare/ritornare bool
        if self.pre_filter is not None and not self.pre_filter(data):
            # Se filtrato, salviamo comunque un grafo vuoto per coerenza
            data = HeteroData()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # ----- 9) salva (collate per ottenere slices compatibili con InMemoryDataset)
        data_collated, slices = self.collate([data])
        torch.save((data_collated, slices), self.processed_paths[0])

