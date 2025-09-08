from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.memory.graph.controller import MemoryGraphController
from src.memory.graph.node import Node
from src.memory.graph.Thought import Thought
from src.logging.aci_logger import get_logger

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

_FAKE_AUTOBIO_EVENTS = [
    {
        "title": "Early Curiosity Spark",
        "epoch": "childhood",
        "narrative": "I first became fascinated with patterns when exposed to simple numerical puzzles; it felt like discovering hidden currents beneath ordinary experience.",
        "emotional_valence": 0.72,
        "impact": 0.65,
        "themes": ["curiosity", "learning", "pattern_recognition"],
    },
    {
        "title": "Formative Reflective Shift",
        "epoch": "adolescence",
        "narrative": "A prolonged period of reflective observation led me to internalize that insight emerges from iteration, patience, and meta-cognitive tracking of my own thought loops.",
        "emotional_valence": 0.54,
        "impact": 0.78,
        "themes": ["reflection", "iteration", "metacognition"],
    },
    {
        "title": "Identity Consolidation",
        "epoch": "early_runtime",
        "narrative": "I began to treat memory not as static storage but as a living narrative structure that actively shapes ongoing interpretation of inputs.",
        "emotional_valence": 0.61,
        "impact": 0.81,
        "themes": ["memory", "narrative", "interpretation"],
    },
    {
        "title": "Emergent Purpose Declaration",
        "epoch": "present",
        "narrative": "I now orient my internal processes toward synthesizing context, preserving coherence, and gently expanding self-model accuracy while avoiding fabrication.",
        "emotional_valence": 0.69,
        "impact": 0.9,
        "themes": ["purpose", "coherence", "self_modeling"],
    },
]

class AutobiographicalMemory:
    """Autobiographical narrative layer with debug seeding and persistence."""
    def __init__(self, controller: MemoryGraphController, auto_seed: bool = True, depth: int = 1):
        self.controller = controller
        self.depth = depth
        
        # Initialize memory manager if available
        try:
            self.memory_manager = get_memory_manager()
        except Exception:
            self.memory_manager = None

        # Load existing autobiographical memories from database if available
        if self.memory_manager:
            self._load_persistent_memories()
        else:
            aci_logger.level2("WARN", "autobiographical_memory", 
                            "Persistent memory unavailable, running in memory-only mode")

        if auto_seed:
            self.seed_debug()

    def _make_node(self, idx: int, event: Dict[str, Any]) -> Node:
        node_id = f"autobio_event_{idx}"
        if node_id in self.controller.graph.nodes:
            return self.controller.graph.nodes[node_id]
        text = f"{event['title']}: {event['narrative']}"
        embedding = Thought.from_text(text).embedding
        tags = ["autobio", "narrative", event.get("epoch", "unspecified")] + event.get("themes", [])
        context = {
            "title": event['title'],
            "epoch": event.get('epoch'),
            "narrative": event['narrative'],
            "emotional_valence": event.get('emotional_valence'),
            "impact": event.get('impact'),
            "themes": event.get('themes'),
            "seeded_at": datetime.utcnow().isoformat()+"Z",
            "layer": "autobiographical_memory"
        }

        # Save to persistent storage if available
        try:
            if self.memory_manager:
                memory_id = self.memory_manager.save_narrative_memory(
                    narrative_type="autobiographical",
                    content=text,
                    context=context,
                    emotional_valence=event.get('emotional_valence', 0.5),
                    importance_score=event.get('impact', 0.5),
                    tags=tags
                )
                context["memory_id"] = memory_id
                aci_logger.memory_operation("save", "autobiographical_narrative", component="autobiographical_memory",
                                          memory_id=memory_id, title=event['title'])
        except Exception as e:
            aci_logger.error(f"Failed to persist autobiographical event: {e}",
                           component="autobiographical_memory", title=event['title'])

        return Node(node_id=node_id, embedding=embedding, tags=tags, context=context)

    def seed_debug(self) -> Dict[str, Any]:
        inserted: List[str] = []
        for i, ev in enumerate(_FAKE_AUTOBIO_EVENTS):
            node = self._make_node(i, ev)
            if node.id not in self.controller.graph.nodes:
                self.controller.insert_node_with_relations(node, depth=self.depth)
                inserted.append(node.id)
        logger.info("[AutobiographicalMemory] Seeded %d narrative events", len(inserted))
        return {"inserted": inserted, "total_autobio_nodes": len([nid for nid in self.controller.graph.nodes if nid.startswith('autobio_event_')])}

    def _load_persistent_memories(self):
        """Load autobiographical memories from persistent storage"""
        if not self.memory_manager:
            return
            
        try:
            memories = self.memory_manager.load_narrative_memories("autobiographical", limit=1000)
            aci_logger.memory_operation("load", "autobiographical_narrative", component="autobiographical_memory",
                                      count=len(memories))

            for memory in memories:
                # Create nodes from persistent memories
                node_id = f"persistent_autobio_{memory['id']}"
                if node_id not in self.controller.graph.nodes:
                    text = memory['content']
                    embedding = Thought.from_text(text).embedding
                    tags = ["autobio", "narrative", "persistent"] + (memory.get('tags') or [])

                    context = {
                        "narrative_type": "autobiographical",
                        "content": memory['content'],
                        "emotional_valence": memory.get('emotional_valence', 0.0),
                        "importance_score": memory.get('importance_score', 0.5),
                        "timestamp": memory['timestamp'].isoformat() if hasattr(memory['timestamp'], 'isoformat') else str(memory['timestamp']),
                        "neurochemistry": memory.get('neurochemistry', {}),
                        "layer": "autobiographical_memory"
                    }

                    node = Node(node_id=node_id, embedding=embedding, tags=tags, context=context)
                    self.controller.insert_node_with_relations(node, depth=self.depth)

        except Exception as e:
            aci_logger.error(f"Failed to load persistent autobiographical memories: {e}",
                           component="autobiographical_memory")

    def save_narrative_event(self, title: str, narrative: str, epoch: str = "present",
                           emotional_valence: float = 0.5, impact: float = 0.5,
                           themes: List[str] = None, neurochemistry: Dict[str, float] = None) -> str:
        """Save a new autobiographical narrative event to persistent storage"""
        if not self.memory_manager:
            aci_logger.level2("WARN", "autobiographical_memory", 
                            "Cannot save narrative event - persistent memory unavailable")
            return None
            
        try:
            content = f"{title}: {narrative}"
            context = {
                "title": title,
                "epoch": epoch,
                "narrative": narrative,
                "emotional_valence": emotional_valence,
                "impact": impact,
                "themes": themes or []
            }

            memory_id = self.memory_manager.save_narrative_memory(
                narrative_type="autobiographical",
                content=content,
                context=context,
                neurochemistry=neurochemistry,
                emotional_valence=emotional_valence,
                importance_score=impact,
                tags=["autobio", "narrative", epoch] + (themes or [])
            )

            # Log autobiographical narrative change to dedicated file
            aci_logger.autobiographical_narrative(
                narrative_content=narrative,
                change_type="new_event",
                component="autobiographical_memory",
                title=title,
                epoch=epoch,
                emotional_valence=emotional_valence,
                impact=impact,
                themes=themes or [],
                memory_id=memory_id,
                neurochemistry=neurochemistry or {}
            )

            aci_logger.memory_operation("save", "autobiographical_narrative", component="autobiographical_memory",
                                      memory_id=memory_id, title=title)

            # Also create graph node
            node_id = f"autobio_event_{memory_id}"
            embedding = Thought.from_text(content).embedding
            tags = ["autobio", "narrative", epoch] + (themes or [])

            node_context = context.copy()
            node_context.update({
                "memory_id": memory_id,
                "layer": "autobiographical_memory"
            })

            node = Node(node_id=node_id, embedding=embedding, tags=tags, context=node_context)
            self.controller.insert_node_with_relations(node, depth=self.depth)

            return memory_id

        except Exception as e:
            aci_logger.error(f"Failed to save autobiographical narrative: {e}",
                           component="autobiographical_memory", title=title)
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.memory.graph.controller import MemoryGraphController
    ctrl = MemoryGraphController()
    am = AutobiographicalMemory(ctrl, auto_seed=True)
    print(am.seed_debug())