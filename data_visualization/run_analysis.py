import os
from pprint import pprint
from pathlib import Path
import torch
from torch_geometric.data import HeteroData

# Adjust sys.path to allow importing modules from the parent directory (Tesi)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.dataset import NeuroDegAnc2VecDataset
from data_visualization.graph_analyzer import GraphAnalyzer

import torch_geometric.transforms as T
from data_pipeline import *


DO_TRANSFORM = True


def main():
    print("--- Starting Graph Data Analysis ---")

    # Define paths relative to the project root
    # Assuming this script is in Tesi/data_visualization
    project_root = Path(__file__).parent.parent
    
    # Ensure the 'data' directory exists for InMemoryDataset processed files
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Paths for the dataset configuration (raw input files)
    c0_nodes_csv = project_root / "networks" / "HD_nodes.csv"
    c1_nodes_csv = project_root / "networks" / "MSA_nodes.csv"
    c0_edges_csv = project_root / "networks" / "HD_edges.csv"
    c1_edges_csv = project_root / "networks" / "MSA_edges.csv"
    anc2vec_npz_path = project_root / "data_pipeline" / "anc2vec_go_embeddings_v1.npz"

    class_labels = (('T1', 'T2'))

    # Check if required raw input files exist before trying to load the dataset
    required_files = [c0_nodes_csv, c1_nodes_csv, c0_edges_csv, c1_edges_csv, anc2vec_npz_path]
    for f_path in required_files:
        if not f_path.exists():
            print(f"Error: Required file not found: {f_path}")
            print("Please ensure all input CSVs and the anc2vec NPZ file are in their correct locations.")
            print("You might need to run 'prepare_network_data.py' to generate network CSVs and ensure 'anc2vec_go_embeddings_v1.npz' is present.")
            return

    # Instantiate the dataset
    try:
        pre_transform = T.Compose([
            MarkCommonNodes(edge_types=class_labels),
            ReindexConsecutive(),
            MergeRelationsToHomogeneous(merge=class_labels, edge_attr_reduce='sum'),
            # FilterSmallConnectedComponents(topk=1),
            FilterLargestComponentPerClass(class_labels=(0, 1))
        ]) if DO_TRANSFORM else None

        dataset = NeuroDegAnc2VecDataset(
            root=str(data_root), # Directory where processed data (hetero.pt) will be stored
            c0_nodes_csv=str(c0_nodes_csv),
            c1_nodes_csv=str(c1_nodes_csv),
            c0_edges_csv=str(c0_edges_csv),
            c1_edges_csv=str(c1_edges_csv),
            anc2vec_npz_path=str(anc2vec_npz_path),
            force_reload=True, # Set to True to re-process if needed, False to load from cache
            class_labels=class_labels,
            transform=None,
            pre_transform=pre_transform
        )
        hetero_data = dataset[0] # Get the first (and likely only) graph from the dataset
        class_labels = dataset.labels # Get the class labels (e.g., ['AD', 'PD']) from the dataset
    except Exception as e:
        print(f"Failed to load NeuroDegAnc2VecDataset: {e}")
        print("Please ensure the dataset can be successfully processed. Check file paths and data integrity.")
        return

    print(f"Loaded.")

    # Instantiate and run the analyzer
    analyzer = GraphAnalyzer(hetero_data, class_labels=class_labels)
    
    node_stats = analyzer.analyze_nodes()
    edge_stats = analyzer.analyze_edges()
    graph_stats = analyzer.analyze_graph()
    
    # Define output path for visualization
    output_viz_dir = project_root / "data_visualization" / "output"
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    viz_output_path = output_viz_dir / "graph_visualization.png"
    analyzer.visualize_graph(output_path=str(viz_output_path))

    print("\n--- Graph Data Analysis Complete ---")
    print("Summary of collected statistics:")

    for k,v in {
        "Node Stats": node_stats,
        "Edge Stats": edge_stats,
        "Graph Stats": graph_stats
    }.items():
        print(k, end=" ")
        pprint(v)

if __name__ == '__main__':
    main()