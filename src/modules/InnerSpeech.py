from __future__ import annotations
from typing import Dict, Any, Optional
import logging

from src.modules.ReasoningLayer import ReasoningLayer

logger = logging.getLogger(__name__)

class InnerSpeech:
    """Generates inner speech (reflective thought) via ReasoningLayer.

    Invoked when PFC selected action is 'reflect'. No heuristic expansion or tone
    synthesis; delegates entirely to the reasoning model with an internal-thinking
    instruction. The output is treated as an internally narrated thought.
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
        pfc = bundle.get('pfc', {})
        action = pfc.get('selected_action') or {}
        if action.get('type') != 'reflect':
            return bundle  # only act on reflect

        # Seed text: prefer action content, else prior thought text, else placeholder
        seed = action.get('content') or bundle.get('current_thought', {}).get('text') or '(empty)'

        # Build minimal context bundle for reasoning layer
        gws_bundle = {
            'scene_description': bundle.get('visual_cortex', {}).get('scene_description') or bundle.get('scene_description') or '(no scene)',
            'neurochemistry': bundle.get('neurochemistry', {}),
            'hippocampus': bundle.get('hippocampus', {}),
        }

        # Instruction to model: generate INNER SPEECH only
        internal_instruction = (
            "INTERNAL REFLECTION: Generate a concise inner speech thought (first-person, present, <=25 words) "
            "purely about current context and seed. Do NOT address a user. Seed: " + seed
        )

        try:
            generated = self.reasoning_layer.reason(internal_instruction, gws_bundle)
        except Exception as e:
            logger.warning("InnerSpeech reasoning failure (%s); using seed fallback", e)
            generated = seed

        # Store inner speech (no heuristic modulation)
        bundle['inner_speech'] = {
            'raw': generated,
            'modulated': generated,  # identical (no extra modulation)
            'seed': seed,
            'model': 'ReasoningLayer'
        }
        return bundle
