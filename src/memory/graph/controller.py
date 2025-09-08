import time
from .memory_graph import MemoryGraph
from .node import Node
import math
from typing import List, Set, Deque, Dict, Any
from collections import deque
import json
import os
from datetime import datetime
from .Thought import Thought


class MemoryGraphController:
    def __init__(self):
        self.graph = MemoryGraph()

    def _to_timestamp(self, ts):
        """Convert timestamp to float for consistent comparison"""
        if isinstance(ts, datetime):
            return ts.timestamp()
        elif isinstance(ts, (int, float)):
            return float(ts)
        else:
            return time.time()  # fallback

    def insert_nodes(self, num_nodes: int):
        for i in range(num_nodes):
            node = Node(node_id=f"n{i}", tags=[f"tag{i%10}"])
            self.graph.add_node(node)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        # Ensure we have valid embedding lists
        if not a or not b or not isinstance(a, list) or not isinstance(b, list):
            return 0.0
        if len(a) != len(b):
            return 0.0
        
        # Ensure all elements are numeric
        try:
            a_nums = [float(x) for x in a]
            b_nums = [float(x) for x in b]
        except (TypeError, ValueError):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a_nums, b_nums))
        na = math.sqrt(sum(x * x for x in a_nums))
        nb = math.sqrt(sum(y * y for y in b_nums))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _compute_relations(self, source: Node, target: Node):
        if source.id == target.id:
            return
        similarity = self._cosine_similarity(source.embedding, target.embedding)
        if similarity > 0:
            self.graph.edges.add_edge(source.id, target.id, "similarity", weight=similarity)
        tag_overlap = len(set(source.tags) & set(target.tags))
        if tag_overlap:
            source_ts = self._to_timestamp(source.timestamp)
            target_ts = self._to_timestamp(target.timestamp)
            time_delta = abs(source_ts - target_ts) + 1e-6
            relevance = tag_overlap / (1.0 + math.log10(time_delta + 1))
            self.graph.edges.add_edge(source.id, target.id, "relevance", weight=relevance)
        
        source_ts = self._to_timestamp(source.timestamp)
        target_ts = self._to_timestamp(target.timestamp)
        if source_ts > target_ts:
            dt = source_ts - target_ts
            causality = 1.0 / (1.0 + dt)
            self.graph.edges.add_edge(source.id, target.id, "causality", weight=causality)

    def _recompute_outgoing(self, node: Node):
        # Remove all existing outgoing edges for this node, then recompute to every other node.
        self.graph.edges.remove_edges(node.id)  # remove all
        for other in self.graph.nodes.values():
            if other.id == node.id:
                continue
            self._compute_relations(node, other)

    def insert_node_with_relations(self, node: Node, depth: int = 2):
        """Insert node, compute relations to existing nodes, then recompute targets' relations to depth."""
        self.graph.add_node(node)
        # First pass: compute edges from new node -> existing
        for other in list(self.graph.nodes.values()):
            if other.id == node.id:
                continue
            self._compute_relations(node, other)
        # Second pass: recompute existing nodes' edges to include new node
        frontier: Deque[str] = deque()
        visited: Set[str] = set()
        # Direct neighbors are all existing nodes (since recomputation is global for correctness)
        for other in self.graph.nodes.values():
            if other.id == node.id:
                continue
            frontier.append(other.id)
            visited.add(other.id)
        current_depth = 1
        while frontier and current_depth <= depth:
            level_size = len(frontier)
            for _ in range(level_size):
                nid = frontier.popleft()
                n = self.graph.nodes[nid]
                self._recompute_outgoing(n)
                # Add neighbors for next depth
                for neigh in self.graph.edges.neighbors(nid):
                    if neigh not in visited and neigh in self.graph.nodes:
                        frontier.append(neigh)
                        visited.add(neigh)
            current_depth += 1

    def test_retrieval_by_tag(self, tag: str):
        return self.graph.get_nodes_by_tag(tag)

    def report(self):
        node_count, edge_count = self.graph.memory_size()
        avg_latency = self.graph.avg_insertion_latency()
        return {"nodes": node_count, "edges": edge_count, "avg_insertion_latency_ms": avg_latency * 1000}

    def seed_from_conversation_json(self, path: str, tag_user: str = "user_msg", tag_assistant: str = "assistant_msg") -> Dict[str, Any]:
        """Seed the memory graph from a JSON conversation file.

        JSON format: list of {"role": "user"|"assistant", "content": "..."
        For each turn we:
          - Create a Thought embedding
          - Create a Node with appropriate tag(s) + role tag
          - Insert with relation computation (depth=1 for speed)
        Returns summary stats.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            convo = json.load(f)
        if not isinstance(convo, list):
            raise ValueError("Conversation JSON must be a list")
        created_ids: List[str] = []
        for idx, turn in enumerate(convo):
            role = (turn.get("role") or "user").lower()
            content = turn.get("content") or ""
            thought = Thought.from_text(content)
            tags = ["conversation", f"turn{idx}"]
            if role == "user":
                tags.append(tag_user)
            else:
                tags.append(tag_assistant)
            node = Node(node_id=f"conv_{idx}", embedding=thought.embedding, tags=tags, context={
                "text": content,
                "role": role,
                "turn_index": idx
            })
            self.insert_node_with_relations(node, depth=1)
            created_ids.append(node.id)
        return {
            "turns": len(created_ids),
            "node_ids": created_ids,
            "memory_size": self.graph.memory_size()
        }
