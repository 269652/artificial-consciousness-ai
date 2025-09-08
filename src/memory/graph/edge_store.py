from typing import Dict, List, Tuple
import random
import time


class EdgeStore:
    def __init__(self):
        # node_id -> List of (neighbor_id, relation_type, weight)
        self.adj_list: Dict[str, List[Tuple[str, str, float]]] = {}
        self.edge_count = 0

    def add_edge(self, from_id: str, to_id: str, relation_type: str, weight: float = 1.0):
        if from_id not in self.adj_list:
            self.adj_list[from_id] = []
        self.adj_list[from_id].append((to_id, relation_type, weight))
        self.edge_count += 1

    def get_edges(self, node_id: str):
        return self.adj_list.get(node_id, [])

    def remove_edges(self, from_id: str, to_id: str | None = None, relation_types: List[str] | None = None):
        """Remove edges matching criteria. Returns number removed."""
        if from_id not in self.adj_list:
            return 0
        before = len(self.adj_list[from_id])
        def keep(e: Tuple[str, str, float]):
            tgt, rel, _ = e
            if to_id is not None and tgt != to_id:
                return True
            if relation_types is not None and rel not in relation_types:
                return True
            if to_id is None and relation_types is None:
                return False  # remove all
            return False
        self.adj_list[from_id] = [e for e in self.adj_list[from_id] if keep(e)]
        removed = before - len(self.adj_list[from_id])
        self.edge_count -= removed
        if not self.adj_list[from_id]:
            del self.adj_list[from_id]
        return removed

    def neighbors(self, node_id: str) -> List[str]:
        return [t for t, _, _ in self.adj_list.get(node_id, [])]

    def size(self):
        return self.edge_count

class SalienceScorer:
    def score(self, node_id: str) -> float:
        # Stub: random or frequency-based
        return random.random()
