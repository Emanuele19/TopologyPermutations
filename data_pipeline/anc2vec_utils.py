# src/transforms/anc2vec_portable.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import numpy as np

def load_anc2vec_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Carica l'NPZ esportato con Python 3.6.
    Ritorna: terms [N], vectors [N,D], go2idx {GO -> idx}
    """
    npz = np.load(path, allow_pickle=False)
    terms = npz["terms"]           # dtype '<U32', shape [N]
    vectors = npz["vectors"]       # dtype 'float32', shape [N, D]
    go2idx = {t: i for i, t in enumerate(terms.tolist())}
    return terms, vectors, go2idx

def merge_embeddings(go_list: Iterable[str], go2idx: Dict[str,int], vectors: np.ndarray) -> np.ndarray:
    """
    Somma i vettori anc2vec dei GO presenti; se vuoto, restituisce zero-vector [D].
    """
    idxs = [go2idx[g.strip()] for g in go_list if g and g.strip() in go2idx]
    if not idxs:
        return np.zeros((vectors.shape[1],), dtype=np.float32)
    return vectors[idxs].sum(axis=0).astype(np.float32)
