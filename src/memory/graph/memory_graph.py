import time
from typing import List, Dict, Optional
from .node import Node
from .edge_store import EdgeStore, SalienceScorer

class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        # Flat chronological sequence preserving first-in insertion order
        # Stores direct Node references so mutations are reflected.
        self.node_sequence: List[Node] = []
        self.edges = EdgeStore()
        self.salience = SalienceScorer()
        self.insertion_times: List[float] = []

    def add_node(self, node: Node):
        start = time.time()
        is_new = node.id not in self.nodes
        self.nodes[node.id] = node
        if is_new:
            self.node_sequence.append(node)
        # If node id already existed we do not duplicate in sequence
        self.insertion_times.append(time.time() - start)

    def add_edge(self, from_id: str, to_id: str, relation_type: str, weight: float = 1.0):
        self.edges.add_edge(from_id, to_id, relation_type, weight=weight)

    def get_nodes_by_tag(self, tag: str) -> List[Node]:
        return [node for node in self.nodes.values() if tag in node.tags]

    # --- New chronological access helpers ---
    def chronological_nodes(self) -> List[Node]:
        """Return all nodes in insertion order (oldest -> newest)."""
        return self.node_sequence

    def recent_nodes(self, n: int) -> List[Node]:
        """Return the n most recently inserted distinct nodes (newest last)."""
        if n <= 0:
            return []
        return self.node_sequence[-n:]

    def iter_reverse_chronological(self):
        """Iterate nodes from newest to oldest."""
        for node in reversed(self.node_sequence):
            yield node

    def find_recent_by_tag(self, tag: str, limit: Optional[int] = None) -> List[Node]:
        """Search in reverse chronological order for nodes with a tag.
        Stops after collecting 'limit' matches (all if limit None)."""
        results: List[Node] = []
        for node in self.iter_reverse_chronological():
            if tag in node.tags:
                results.append(node)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def memory_size(self):
        return len(self.nodes), self.edges.size()

    def avg_insertion_latency(self):
        if not self.insertion_times:
            return 0.0
        return sum(self.insertion_times) / len(self.insertion_times)
