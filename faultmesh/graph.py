"""Graph module for FaultMesh AI.

Responsibility:
    Build and manage the IoT sensor network topology using NetworkX,
    providing graph traversal, neighborhood queries, and centrality analysis.
"""

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx


DEFAULT_EDGES: List[Tuple[str, str]] = [
    ("S3", "S1"),
    ("S3", "S2"),
    ("S3", "S4"),
    ("S1", "S2"),
    ("S4", "S5"),
    ("S2", "S5"),
    ("S3", "S5"),
]

DEFAULT_NODES: List[str] = ["S1", "S2", "S3", "S4", "S5"]


class SensorNetworkGraph:
    """Manages the topological graph representation of the sensor network."""

    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize the sensor network graph.

        Args:
            nodes: List of sensor node names (defaults to S1..S5).
            edges: List of undirected sensor connection tuples.
        """
        self.nodes = nodes or DEFAULT_NODES.copy()
        self.edges = edges or DEFAULT_EDGES.copy()
        self.graph = nx.Graph()
        self.build_graph()

    def build_graph(self) -> None:
        """Construct the NetworkX graph with nodes and physical coupling edges."""
        # TODO: Add node attributes (baseline, parameter name, units) and build edges
        self.graph.clear()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def get_neighbors(self, node: str) -> List[str]:
        """Return list of directly adjacent neighboring sensors for a node.

        Args:
            node: Target sensor ID.

        Returns:
            List of neighbor sensor IDs.
        """
        if node not in self.graph:
            return []
        return list(self.graph.neighbors(node))

    def get_degree_centrality(self) -> Dict[str, float]:
        """Calculate normalized degree centrality for all nodes in the network.

        Returns:
            Dict mapping sensor ID to degree centrality score.
        """
        # TODO: Return networkx degree centrality
        return nx.degree_centrality(self.graph)

    def get_betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for all nodes in the network.

        Returns:
            Dict mapping sensor ID to betweenness centrality score.
        """
        # TODO: Return networkx betweenness centrality
        return nx.betweenness_centrality(self.graph)

    def get_coverage_score(self, node: str) -> float:
        """Compute the topological coverage/influence score of a node in the mesh.

        Args:
            node: Target sensor ID.

        Returns:
            Coverage score in [0, 1] based on degree and neighborhood size.
        """
        # TODO: Combine degree centrality and 2-hop reachability into coverage score
        if node not in self.graph:
            return 0.0
        degree = self.graph.degree(node)
        max_possible = len(self.nodes) - 1
        return float(degree) / float(max_possible) if max_possible > 0 else 0.0

    def get_adjacency_dict(self) -> Dict[str, List[str]]:
        """Return adjacency map for all nodes in the network.

        Returns:
            Dict mapping each sensor ID to its list of neighbors.
        """
        return {node: self.get_neighbors(node) for node in self.nodes}
