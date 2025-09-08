from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import logging
import math
import numpy as np
import json

from src.memory.graph.controller import MemoryGraphController
from src.memory.graph.node import Node
from src.memory.graph.Thought import Thought
from src.logging.aci_logger import get_logger, LogLevel

# Try to import persistent memory, handle gracefully if unavailable
try:
    from src.memory.persistent_memory import get_memory_manager
    _PERSISTENT_MEMORY_AVAILABLE = True
except ImportError:
    _PERSISTENT_MEMORY_AVAILABLE = False
    def get_memory_manager():
        return None

logger = logging.getLogger(__name__)
aci_logger = get_logger()

def _sanitize_for_json(obj):
    """Convert numpy arrays and other non-JSON-serializable objects to JSON-safe format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj

class Hippocampus:
    """Hippocampus module (retrieval-focused) with persistent episodic memory.

    Instead of creating new memory nodes, it searches an existing global
    memory graph (provided in the bundle) for nodes similar to the current
    thought embedding.

    It also reconstructs a short recent conversation window from the
    latest N episodic/thought nodes (chronological order).
    """

    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.25, recent_k: int = 20):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.recent_k = recent_k  # how many latest memories to pull for conversation rebuild
        try:
            self.memory_manager = get_memory_manager()
        except Exception:
            self.memory_manager = None
        self.episode_counter = 0
        self._pending_episodic_nodes = []  # Store nodes until controller is available

        # Load existing episodic memories from database
        if self.memory_manager:
            self._load_persistent_episodic_memories()
        else:
            aci_logger.level2(LogLevel.WARN, "hippocampus", 
                            "Persistent memory unavailable, running in memory-only mode")

    # --- Internal helpers ---
    def _ensure_memory_graph(self, bundle: Dict[str, Any]) -> MemoryGraphController:
        controller: Optional[MemoryGraphController] = bundle.get("memory_graph_controller")  # type: ignore
        if controller is None:
            controller = MemoryGraphController()
            bundle["memory_graph_controller"] = controller

        # Store controller reference for later use
        self._controller = controller

        # Add any pending episodic nodes to the graph
        if hasattr(self, '_pending_episodic_nodes'):
            for node in self._pending_episodic_nodes:
                if node.id not in controller.graph.nodes:
                    controller.insert_node_with_relations(node, depth=1)
            self._pending_episodic_nodes = []

        return controller

    def _embed(self, text: str) -> List[float]:
        return Thought.from_text(text).embedding

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _role_from_tags(self, tags: List[str]) -> str:
        if 'user_msg' in tags:
            return 'user'
        if 'assistant_msg' in tags:
            return 'assistant'
        if 'thought' in tags:
            return 'thought'
        return 'memory'

    def _reconstruct_recent_conversation(self, controller: MemoryGraphController) -> Dict[str, Any]:
        graph = controller.graph
        sequence: List[Node] = graph.node_sequence  # chronological oldest->newest
        if not sequence:
            return {"recent_node_ids": [], "turns": [], "transcript": ""}
        # Start from latest (newest) node
        start = sequence[-1]
        chain: List[Node] = [start]
        current = start
        steps = 1
        # Follow outgoing 'conversation' edges iteratively (assumed: new node -> previous node)
        while steps < self.recent_k:
            conv_edges = [e for e in graph.edges.get_edges(current.id) if e[1] == "conversation"]
            if not conv_edges:
                break
            # Pick highest weight edge (most salient connection)
            conv_edges.sort(key=lambda x: x[2], reverse=True)  # (to_id, rel, weight)
            next_id = conv_edges[0][0]
            if any(n.id == next_id for n in chain):  # cycle guard
                break
            next_node = graph.nodes.get(next_id)
            if not next_node:
                break
            chain.append(next_node)
            current = next_node
            steps += 1
        # chain currently newest -> oldest; reverse for chronological order
        ordered = list(reversed(chain))
        turns: List[Dict[str, Any]] = []
        for n in ordered:
            if not any(t in n.tags for t in ("thought", "conversation", "user_msg", "assistant_msg")):
                continue
            text = n.context.get("text") or n.context.get("thought") or None
            answer = None
            sel = n.context.get("selected_action") if isinstance(n.context, dict) else None
            if isinstance(sel, dict):
                answer = sel.get("content")
            turns.append({
                "node_id": n.id,
                "role": self._role_from_tags(n.tags),
                "text": text,
                "answer": answer
            })
        transcript_lines: List[str] = []
        for t in turns:
            if t.get("answer"):
                transcript_lines.append(f"User: {t['text']}\nAI: {t['answer']}")
            else:
                transcript_lines.append(f"{t['role']}: {t['text']}")
        transcript = "\n".join(transcript_lines)
        # print(f"Transscript: {transcript}")
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[Hippocampus] Conversation edge-walk | nodes_traversed=%d start=%s end=%s", 
                len(chain),
                ordered[0].id if ordered else None,
                ordered[-1].id if ordered else None,
            )
        return {
            "recent_node_ids": [n.id for n in ordered],
            "turns": turns,
            "transcript": transcript,
        }

    def _load_persistent_episodic_memories(self):
        """Load episodic memories from persistent storage and store as pending nodes"""
        if not self.memory_manager:
            return
            
        try:
            episodes = self.memory_manager.load_episodic_memories(limit=1000)
            aci_logger.memory_operation("load", "episodic_memory", component="hippocampus",
                                      count=len(episodes))

            for episode in episodes:
                # Create graph nodes from episodic memories
                node_id = f"episodic_{episode['episode_id']}_{episode['sequence_number']}"
                
                embedding = episode.get('embedding')
                if not embedding:
                    # Generate embedding if not stored
                    embedding = Thought.from_text(episode['content']).embedding

                tags = ["episodic", "memory"] + (episode.get('tags') or [])
                context = {
                    "episode_id": episode['episode_id'],
                    "sequence_number": episode['sequence_number'],
                    "content": episode['content'],
                    "sensory_data": episode.get('sensory_data', {}),
                    "neurochemistry": episode.get('neurochemistry', {}),
                    "emotional_context": episode.get('emotional_context', {}),
                    "spatial_context": episode.get('spatial_context', {}),
                    "social_context": episode.get('social_context', {}),
                    "thought_chain": episode.get('thought_chain', []),
                    "timestamp": episode['timestamp'].isoformat() if hasattr(episode['timestamp'], 'isoformat') else str(episode['timestamp']),
                    "layer": "episodic_memory"
                }

                node = Node(node_id=node_id, embedding=embedding, tags=tags, context=context)
                # Store as pending until controller is available
                self._pending_episodic_nodes.append(node)

        except Exception as e:
            aci_logger.error(f"Failed to load persistent episodic memories: {e}",
                           component="hippocampus")

    def save_episodic_memory(self, episode_id: str, content: str,
                           sensory_data: Dict[str, Any] = None,
                           neurochemistry: Dict[str, float] = None,
                           emotional_context: Dict[str, Any] = None,
                           spatial_context: Dict[str, Any] = None,
                           social_context: Dict[str, Any] = None,
                           thought_chain: List[str] = None,
                           embedding: List[float] = None,
                           tags: List[str] = None) -> str:
        """Save an episodic memory to persistent storage"""
        try:
            if not embedding:
                embedding = Thought.from_text(content).embedding

            sequence_number = self.episode_counter
            self.episode_counter += 1

            memory_id = self.memory_manager.save_episodic_memory(
                episode_id=episode_id,
                sequence_number=sequence_number,
                content=content,
                sensory_data=_sanitize_for_json(sensory_data),
                neurochemistry=_sanitize_for_json(neurochemistry),
                emotional_context=_sanitize_for_json(emotional_context),
                spatial_context=_sanitize_for_json(spatial_context),
                social_context=_sanitize_for_json(social_context),
                thought_chain=_sanitize_for_json(thought_chain),
                embedding=_sanitize_for_json(embedding),
                tags=tags
            )

            aci_logger.memory_operation("save", "episodic_memory", component="hippocampus",
                                      memory_id=memory_id, episode_id=episode_id)

            # Also create graph node
            controller = getattr(self, '_controller', None)
            if controller:
                node_id = f"episodic_{episode_id}_{sequence_number}"
                node_tags = ["episodic", "memory"] + (tags or [])

                context = {
                    "episode_id": episode_id,
                    "sequence_number": sequence_number,
                    "content": content,
                    "sensory_data": _sanitize_for_json(sensory_data or {}),
                    "neurochemistry": _sanitize_for_json(neurochemistry or {}),
                    "emotional_context": _sanitize_for_json(emotional_context or {}),
                    "spatial_context": _sanitize_for_json(spatial_context or {}),
                    "social_context": _sanitize_for_json(social_context or {}),
                    "thought_chain": _sanitize_for_json(thought_chain or []),
                    "memory_id": memory_id,
                    "layer": "episodic_memory"
                }

                node = Node(node_id=node_id, embedding=embedding, tags=node_tags, context=context)
                controller.insert_node_with_relations(node, depth=1)

            return memory_id

        except Exception as e:
            aci_logger.error(f"Failed to save episodic memory: {e}",
                           component="hippocampus", episode_id=episode_id)
            return None

    def process(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        thought_info = bundle.get("current_thought") or {}
        thought_text: str = thought_info.get("text") or "(empty thought)"
        embedding: List[float] = thought_info.get("embedding") or self._embed(thought_text)

        controller = self._ensure_memory_graph(bundle)
        graph = controller.graph

        # Save episodic memory for this thought
        episode_id = f"session_{getattr(self.memory_manager, 'session_id', 'unknown')}"
        self.save_episodic_memory(
            episode_id=episode_id,
            content=thought_text,
            sensory_data=bundle.get('sensory_data', {}),
            neurochemistry=bundle.get('neurochemistry', {}),
            emotional_context=bundle.get('emotional_context', {}),
            spatial_context=bundle.get('spatial_context', {}),
            social_context=bundle.get('social_context', {}),
            thought_chain=bundle.get('thought_chain', []),
            embedding=embedding,
            tags=["thought", "current"]
        )

        # Gather similarities
        similarities: List[Tuple[str, float, Node]] = []
        for node_id, node in graph.nodes.items():
            if not node.embedding:
                continue
            sim = self._cosine(embedding, node.embedding)
            if sim >= self.similarity_threshold:
                similarities.append((node_id, sim, node))

        # Rank & trim
        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[: self.top_k]

        if logger.isEnabledFor(logging.INFO):
            top_debug = [f"{nid}:{sim:.3f}" for (nid, sim, _n) in top]
            logger.info(
                "[Hippocampus] Retrieval summary | searched=%d passed_threshold=%d top_k=%d sims=%s threshold=%.2f",
                len(graph.nodes),
                len(similarities),
                len(top),
                ','.join(top_debug),
                self.similarity_threshold,
            )

        matches = [
            {
                "node_id": nid,
                "similarity": round(sim, 4),
                "text": n.context.get("text") or n.context.get("thought") or None,
                "tags": n.tags,
            }
            for (nid, sim, n) in top
        ]

        thoughts = [
            n.context.get("text") or n.context.get("thought") or None
            for (nid, sim, n) in top

        ]

        # Recent conversation reconstruction
        convo = self._reconstruct_recent_conversation(controller)

        bundle.setdefault("hippocampus", {})
        bundle["hippocampus"].update(
            {
                "matches": matches,
                "thoughts": thoughts,
                "searched_count": len(graph.nodes),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k,
                # "recent_conversation": convo,
                "recent_k": self.recent_k,
            }
        )
        return bundle
