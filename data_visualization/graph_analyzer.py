from __future__ import annotations
import torch
from torch_geometric.data import HeteroData, Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

class GraphAnalyzer:
    """
    Performs data analysis and visualization on a torch_geometric.data.HeteroData object.
    """
    def __init__(self, data: HeteroData | Data, class_labels: Optional[List[str]] = None):
        if not isinstance(data, (HeteroData, Data)):
            raise TypeError("Input data must be a torch_geometric.data.HeteroData or Data object.")
        self.data = data
        self.is_hetero = isinstance(data, HeteroData)
        self.node_type = 'protein' # Assuming 'protein' is the main node type
        self.num_nodes = self.data[self.node_type].num_nodes if self.is_hetero else self.data.num_nodes
        self.class_labels = class_labels if class_labels is not None else ['AD', 'PD'] # Default if not provided

    def analyze_nodes(self) -> Dict[str, Any]:
        """
        Analyzes node-level statistics.
        """
        print("\n--- Node-Level Statistics ---")
        node_store = self.data[self.node_type] if self.is_hetero else self.data
        
        num_nodes = node_store.num_nodes
        print(f"Total number of nodes: {num_nodes}")

        labels = node_store.y.cpu().numpy()
        label_counts = Counter(labels)
        
        label_0_name = self.class_labels[0] if len(self.class_labels) > 0 else 'Unknown'
        label_1_name = self.class_labels[1] if len(self.class_labels) > 1 else 'Unknown'
        print(f"Node label distribution (0: {label_0_name}, 1: {label_1_name}, 2: Common):")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} nodes ({count/num_nodes:.2%})")

        is_common_present = hasattr(node_store, 'is_common')
        common_nodes_count = node_store.is_common.sum().item() if is_common_present else 0
        print(f"Nodes marked as 'is_common': {common_nodes_count} ({common_nodes_count/num_nodes:.2%})" if is_common_present else "No 'is_common' attribute found.")

        feature_dim = node_store.x.shape[1] if hasattr(node_store, 'x') else 0
        print(f"Node feature dimension (x): {feature_dim}")
        
        string_ids_present = hasattr(node_store, 'string_id')
        print(f"String IDs present: {string_ids_present}")

        return {
            "num_nodes": num_nodes,
            "label_distribution": label_counts,
            "common_nodes_count": common_nodes_count,
            "feature_dimension": feature_dim,
            "string_ids_present": string_ids_present
        }

    def analyze_edges(self) -> Dict[str, Any]:
        """
        Analyzes edge-level statistics for each relation type.
        """
        print("\n--- Edge-Level Statistics ---")
        edge_stats = {}
        
        if self.is_hetero:
            for key in self.data.edge_types:
                src_type, rel_type, dst_type = key
                if src_type != self.node_type or dst_type != self.node_type:
                    continue

                edge_store = self.data[key]
                num_edges = edge_store.edge_index.size(1)
                print(f"Relation '{rel_type}':")
                print(f"  Number of edges: {num_edges}")
                
                avg_degree = num_edges / self.num_nodes if self.num_nodes > 0 else 0
                
                attr_stats = self._get_attr_stats(edge_store)
                if attr_stats:
                    print(f"  Edge attribute ('edge_attr') statistics: {attr_stats}")
                
                edge_stats[rel_type] = {
                    'num_edges': num_edges,
                    'avg_degree': avg_degree,
                    'edge_attr_stats': attr_stats
                }
        else:
            # Homogeneous case
            num_edges = self.data.edge_index.size(1)
            print(f"Homogeneous Edges (Merged):")
            print(f"  Number of edges: {num_edges}")
            avg_degree = num_edges / self.num_nodes if self.num_nodes > 0 else 0
            attr_stats = self._get_attr_stats(self.data)
            if attr_stats:
                print(f"  Edge attribute ('edge_attr') statistics: {attr_stats}")
            edge_stats['merged'] = {
                'num_edges': num_edges,
                'avg_degree': avg_degree,
                'edge_attr_stats': attr_stats
            }
            
        return edge_stats

    def _get_attr_stats(self, store) -> Dict[str, float]:
        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
            edge_attr = store.edge_attr.cpu().numpy()
            return {
                'min': float(np.min(edge_attr)),
                'max': float(np.max(edge_attr)),
                'mean': float(np.mean(edge_attr)),
                'std': float(np.std(edge_attr))
            }
        return {}

    def analyze_graph(self) -> Dict[str, Any]:
        """
        Analyzes graph-level statistics by combining all relations into a single homogeneous graph.
        """
        print("\n--- Graph-Level Statistics (Homogeneous View) ---")
        
        if self.is_hetero:
            combined_edge_indices = []
            for key in self.data.edge_types:
                src_type, rel_type, dst_type = key
                if src_type == self.node_type and dst_type == self.node_type:
                    combined_edge_indices.append(self.data[key].edge_index)
            all_edges_tensor = torch.cat(combined_edge_indices, dim=1) if combined_edge_indices else None
        else:
            all_edges_tensor = self.data.edge_index
        
        if all_edges_tensor is None or all_edges_tensor.numel() == 0:
            print("No edges found for the main node type. Cannot compute graph-level statistics.")
            return {
                "num_connected_components": 0,
                "density": 0.0,
                "largest_component_size": 0,
                "num_isolated_nodes": self.num_nodes
            }

        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        
        edges_list = all_edges_tensor.cpu().numpy().T
        G.add_edges_from(edges_list)
        
        G.remove_edges_from(nx.selfloop_edges(G))

        num_connected_components = nx.number_connected_components(G)
        print(f"Number of connected components: {num_connected_components}")

        density = nx.density(G)
        print(f"Graph density: {density:.4f}")

        component_sizes = [len(c) for c in nx.connected_components(G)]
        largest_component_size = max(component_sizes) if component_sizes else 0
        print(f"Size of the largest connected component: {largest_component_size}")
        
        num_isolated_nodes = sum(1 for node in G.nodes() if G.degree[node] == 0)
        print(f"Number of isolated nodes: {num_isolated_nodes}")

        return {
            "num_connected_components": num_connected_components,
            "density": density,
            "largest_component_size": largest_component_size,
            "num_isolated_nodes": num_isolated_nodes
        }

    def visualize_graph(self, output_path: str = "graph_visualization.png", max_nodes_for_plot: int = 500):
        """
        Visualizes the graph structure.
        For large graphs, this might be slow or unreadable.
        """
        print(f"\n--- Graph Visualization (saving to {output_path}) ---")
        
        if self.num_nodes > max_nodes_for_plot:
            print(f"Warning: Graph has {self.num_nodes} nodes, which is > {max_nodes_for_plot}. "
                  "Visualization might be slow or unreadable. Skipping full plot.")
            print("Consider sampling a subgraph or increasing 'max_nodes_for_plot'.")
            return

        if self.is_hetero:
            combined_edge_indices = []
            for key in self.data.edge_types:
                src_type, rel_type, dst_type = key
                if src_type == self.node_type and dst_type == self.node_type:
                    combined_edge_indices.append(self.data[key].edge_index)
            all_edges_tensor = torch.cat(combined_edge_indices, dim=1) if combined_edge_indices else None
        else:
            all_edges_tensor = self.data.edge_index
        
        if all_edges_tensor is None or all_edges_tensor.numel() == 0:
            print("No edges found for visualization.")
            return

        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        edges_list = all_edges_tensor.cpu().numpy().T
        G.add_edges_from(edges_list)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Node colors based on 'y' labels
        node_colors = []
        node_store = self.data[self.node_type] if self.is_hetero else self.data
        labels = node_store.y.cpu().numpy()
        color_map = {
            0: 'skyblue',  # Corresponds to self.class_labels[0]
            1: 'lightcoral', # Corresponds to self.class_labels[1]
            2: 'lightgreen' # Common nodes
        }
        for node_id in range(self.num_nodes):
            node_colors.append(color_map.get(labels[node_id], 'gray'))

        # Node shapes/sizes based on 'is_common'
        node_shapes = []
        node_sizes = []
        is_common_present = hasattr(node_store, 'is_common')
        if is_common_present:
            is_common_mask = node_store.is_common.cpu().numpy()
            for node_id in range(self.num_nodes):
                if is_common_mask[node_id]:
                    node_shapes.append('s') # Square for common nodes
                    node_sizes.append(100)
                else:
                    node_shapes.append('o') # Circle for non-common nodes
                    node_sizes.append(50)
        else:
            node_shapes = ['o'] * self.num_nodes
            node_sizes = [50] * self.num_nodes

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, iterations=50)

        unique_shapes = set(node_shapes)
        for shape in unique_shapes:
            nodes_with_shape = [i for i, s in enumerate(node_shapes) if s == shape]
            colors_for_shape = [node_colors[i] for i in nodes_with_shape]
            sizes_for_shape = [node_sizes[i] for i in nodes_with_shape]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_shape, node_color=colors_for_shape, node_shape=shape, node_size=sizes_for_shape, alpha=0.8)

        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Label 0 ({self.class_labels[0]})', markerfacecolor=color_map.get(0, 'gray'), markersize=10))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Label 1 ({self.class_labels[1]})', markerfacecolor=color_map.get(1, 'gray'), markersize=10))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Label 2 (Common)', markerfacecolor=color_map.get(2, 'gray'), markersize=10))
        if is_common_present:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label='Node was Common', markerfacecolor='black', markersize=10))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Node not Common', markerfacecolor='black', markersize=10))

        plt.legend(handles=legend_elements, loc='best')
        plt.title("Graph Visualization (Nodes colored by label, shape by 'is_common')")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Graph visualization saved to {output_path}")

if __name__ == '__main__':
    print("Creating a dummy HeteroData object for demonstration...")
    dummy_data = HeteroData()
    
    dummy_data['protein'].x = torch.randn(10, 16)
    dummy_data['protein'].y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 0, 1], dtype=torch.long)
    dummy_data['protein'].is_common = torch.tensor([False, False, True, False, False, True, False, False, False, False], dtype=torch.bool)
    dummy_data['protein'].string_id = [f'P{i}' for i in range(10)]

    dummy_data[('protein', 'AD', 'protein')].edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6]
    ], dtype=torch.long)
    dummy_data[('protein', 'AD', 'protein')].edge_attr = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.85, 0.75, 0.95, 0.65], dtype=torch.float).reshape(-1, 1)

    dummy_data[('protein', 'PD', 'protein')].edge_index = torch.tensor([
        [2, 3, 4, 5, 8, 9],
        [3, 2, 5, 4, 9, 8]
    ], dtype=torch.long)
    dummy_data[('protein', 'PD', 'protein')].edge_attr = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.55, 0.65], dtype=torch.float).reshape(-1, 1)

    analyzer = GraphAnalyzer(dummy_data, class_labels=['AD', 'PD'])
    
    node_stats = analyzer.analyze_nodes()
    edge_stats = analyzer.analyze_edges()
    graph_stats = analyzer.analyze_graph()
    analyzer.visualize_graph(output_path="dummy_graph_visualization.png")

    print("\n--- Analysis Complete ---")
    print("Node Statistics:", node_stats)
    print("Edge Statistics:", edge_stats)
    print("Graph Statistics:", graph_stats)