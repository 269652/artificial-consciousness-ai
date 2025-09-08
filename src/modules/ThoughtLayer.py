from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
from dataclasses import asdict

from src.modules.ReasoningLayer import ReasoningLayer  # Uses existing reasoning pipeline
from src.memory.graph.Thought import Thought  # For embedding / representation

logger = logging.getLogger(__name__)

class ThoughtLayer:
    """Compiles the current global workspace bundle into a Thought.

    Raw input precedence now:
      1. Explicit external user input for this step (last_input)
      2. Inner speech modulated content (if present)
      3. Fallback placeholder
    """
    def __init__(self, reasoning_layer: Optional[ReasoningLayer] = None):
        if reasoning_layer is None:
            try:
                from src.config.reasoning_config import create_reasoning_layer
                self.reasoning_layer = create_reasoning_layer()
            except ImportError:
                # Fallback to default Perplexity
                self.reasoning_layer = ReasoningLayer()
        else:
            self.reasoning_layer = reasoning_layer

    def process(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        inner = bundle.get('inner_speech', {})
        # Determine raw input strictly from user or inner speech
        if bundle.get('last_input'):
            raw_input = bundle['last_input']
        elif inner.get('modulated'):
            raw_input = inner.get('modulated')
        elif inner.get('raw'):
            raw_input = inner.get('raw')
        else:
            raw_input = '(no explicit input)'

        scene_description = (
            bundle.get("scene_description") or
            bundle.get("visual_cortex", {}).get("scene_description") or
            bundle.get("sensory_data", {}).get("environmental", {}).get("terrain", {}).get("type") or
            "(no scene description)"
        )
        gws_bundle = {
            "scene_description": scene_description,
            "neurochemistry": bundle.get("neurochemistry", {}),
            "hippocampus": bundle.get("hippocampus", {}),
            "inner_speech": bundle.get("inner_speech", {}),
            "self_model": bundle.get("self_model", {}),
            "autobiographical_memory": bundle.get("autobiographical_memory", {})
        }

        try:
            prompt = (
                "You are currently reflecting on your own inner speech thuoghts:"
                f"{raw_input}\n"
            )
            reasoning_result = self.reasoning_layer.reason(raw_input, gws_bundle)
            reasoning_dict = asdict(reasoning_result)
        except Exception as e:
            logger.warning("Reasoning failed (%s); using heuristic fallback", e)
            reasoning_dict = {
                "raw_input": raw_input,
                "cleaned_input": raw_input.strip(),
                "reasoned_input": raw_input.strip(),
                "scene_description": scene_description,
                "reasoning_modulator": "neutral",
                "prompt": f"Raw: {raw_input}\nScene: {scene_description}"
            }

        embedding_text = reasoning_dict.get("reasoned_input") or reasoning_dict.get("cleaned_input") or raw_input
        thought_obj = Thought.from_text(embedding_text)

        bundle["reasoning_result"] = reasoning_result
        bundle["current_thought"] = {
            "text": embedding_text,
            "embedding": thought_obj.embedding,
            "source_raw_input": raw_input,
            "scene_description": scene_description
        }
        return bundle
