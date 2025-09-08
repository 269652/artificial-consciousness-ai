import time
from typing import Any, Dict, List, Optional

class Node:
    def __init__(self, node_id: str, embedding: Optional[List[float]] = None, tags: Optional[List[str]] = None,
                 timestamp: Optional[float] = None, context: Optional[Dict[str, Any]] = None,
                 thought_chain: Optional[List[str]] = None, neurochemistry: Optional[Dict[str, float]] = None):
        self.id = node_id
        self.embedding = embedding if embedding is not None else []  # Placeholder / latent vector
        self.tags = tags if tags is not None else []
        self.timestamp = timestamp if timestamp is not None else time.time()
        # New extended attributes
        self.context: Dict[str, Any] = context if context is not None else {}
        self.thought_chain: List[str] = thought_chain if thought_chain is not None else []
        self.neurochemistry: Dict[str, float] = neurochemistry if neurochemistry is not None else {}
        self.read_count: int = 0
        self.merge_count: int = 0

    # --- Convenience methods ---
    def increment_read(self, n: int = 1):
        self.read_count += n

    def increment_merge(self, n: int = 1):
        self.merge_count += n

    def append_thought(self, thought: str):
        self.thought_chain.append(thought)
        # Update embedding with new thought chain
        if self.thought_chain:
            from src.memory.graph.Thought import Thought
            updated_thought = Thought.from_texts(self.thought_chain)
            self.embedding = updated_thought.embedding

    def update_neurochemistry(self, deltas: Dict[str, float]):
        for k, v in deltas.items():
            self.neurochemistry[k] = self.neurochemistry.get(k, 0.0) + v

    def __repr__(self):
        return ("Node(id={id}, tags={tags}, timestamp={ts}, reads={reads}, merges={merges})".format(
            id=self.id, tags=self.tags, ts=self.timestamp, reads=self.read_count, merges=self.merge_count
        ))
