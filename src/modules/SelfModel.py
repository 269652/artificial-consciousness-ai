from __future__ import annotations
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os, sys

# Allow running this file directly: adjust sys.path to project root so 'src' can be imported
if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.memory.graph.controller import MemoryGraphController
from src.memory.graph.node import Node
from src.memory.graph.Thought import Thought
from src.logging.aci_logger import get_logger

logger = logging.getLogger(__name__)
aci_logger = get_logger()

_PERSONA_NODE_ID = "self_model_persona"

_DEFAULT_PERSONALITY_TEXT = (
    "I am an experimental cognitive system oriented toward clarity, reflective self-monitoring, "
    "coherence maintenance, and ethical minimalism (I avoid fabrication and surface uncertainty)."
)

class SelfModelLayer:
    """Minimal self model: maintains a single evolving personality narrative text."""
    def __init__(self, controller: MemoryGraphController, auto_seed: bool = True):
        self.controller = controller
        if auto_seed:
            self.ensure_persona()

    def ensure_persona(self) -> Node:
        if _PERSONA_NODE_ID in self.controller.graph.nodes:
            return self.controller.graph.nodes[_PERSONA_NODE_ID]
        embedding = Thought.from_text(_DEFAULT_PERSONALITY_TEXT).embedding
        node = Node(
            node_id=_PERSONA_NODE_ID,
            embedding=embedding,
            tags=["self_model", "persona"],
            context={
                "persona_text": _DEFAULT_PERSONALITY_TEXT,
                "created_at": datetime.utcnow().isoformat()+"Z",
                "last_update": datetime.utcnow().isoformat()+"Z",
                "layer": "self_model"
            }
        )
        self.controller.insert_node_with_relations(node, depth=0)
        # logger.info("[SelfModel] Created initial persona node")
        return node

    def update_personality(self, new_text: str, append: bool = False, max_len: int = 800) -> Dict[str, Any]:
        node = self.ensure_persona()
        old_text = node.context.get('persona_text', '')
        
        if append:
            combined = f"{old_text} {new_text}".strip()
        else:
            combined = new_text.strip()
        if len(combined) > max_len:
            combined = combined[:max_len].rstrip()
        
        # Only log if there's a meaningful change
        if combined != old_text:
            personality_changes = {
                "old_text": old_text,
                "new_text": combined,
                "change_method": "append" if append else "replace",
                "text_length": len(combined)
            }
            
            # Log personality changes to dedicated file
            aci_logger.personality_self_model(
                personality_changes=personality_changes,
                change_type="personality_update",
                component="self_model",
                append_mode=append,
                text_length_change=len(combined) - len(old_text)
            )
        
        node.context['persona_text'] = combined
        node.context['last_update'] = datetime.utcnow().isoformat()+"Z"
        node.embedding = Thought.from_text(combined).embedding
        # logger.info("[SelfModel] Updated personality text (len=%d)", len(combined))
        return {"persona_text": combined, "last_update": node.context['last_update']}

    def export_persona(self) -> str:
        node = self.ensure_persona()
        return node.context.get('persona_text','')
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctrl = MemoryGraphController()
    sm = SelfModelLayer(ctrl, auto_seed=True)
    # print(sm.export_persona())