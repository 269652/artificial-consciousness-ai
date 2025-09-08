from __future__ import annotations
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PrefrontalCortex:
    """Prefrontal Cortex module.

    Responsibilities (scaffold):
      - Integrate hippocampal retrieval (context matches) + current reasoning result.
      - Form simple action plans: 'speak' (external) or 'reflect' (internal meta-cognition when no new input).
      - Future: arbitration across multiple possible actions (move, attend, store, query, etc.).

    Output appended to bundle under keys:
      bundle['pfc'] = { 'selected_action': {...}, 'candidate_actions': [...], 'rationale': str }
      bundle['cognitive_actions'] = list of action dicts (mirrors candidate_actions for now)
    """

    def __init__(self):
        pass

    def _build_speak_action(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        content = bundle.get('reasoning_result') or bundle.get('current_thought', {}).get('text') or ''
        length_factor = min(len(str(content)) / 60.0, 1.0)
        confidence = 0.4 + 0.6 * length_factor
        return {
            'type': 'speak',
            'modality': 'speech',
            'content': content,
            'confidence': round(confidence, 3),
            'source': 'PFC',
        }

    def _build_reflect_action(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        ct = bundle.get('current_thought', {})
        base_text = ct.get('text') or ct.get('cleaned_input') or '(empty)'
        
        # Check if we're in mind wandering mode
        simulation_phase = bundle.get('simulation_phase', 'conversation')
        
        if simulation_phase == 'mind_wander':
            # During mind wandering, focus on memories and past experiences
            hippocampus = bundle.get('hippocampus', {})
            matches = hippocampus.get('matches', [])
            if matches:
                # Use the most similar memory for reflection
                top_match = matches[0]
                memory_text = top_match.get('text', '')
                if memory_text:
                    base_text = f"Reflecting on past memory: {memory_text[:100]}..."
        
        # Heuristic introspection depth influenced by hypothetical neurochemistry if present
        neuro = bundle.get('neurochemistry') or {}
        serotonin = neuro.get('serotonin', 0.5)
        dopamine = neuro.get('dopamine', 0.5)
        ne = neuro.get('ne', 0.5)
        depth_score = (serotonin + dopamine + (1 - ne)) / 3.0  # calmer + motivated + not hypervigilant
        confidence = round(0.5 + 0.4 * depth_score, 3)
        return {
            'type': 'reflect',
            'modality': 'internal',
            'content': base_text,
            'confidence': confidence,
            'source': 'PFC',
            'meta': {
                'depth_score': depth_score,
                'phase': simulation_phase
            }
        }

    def _build_navigate_action(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic: if input mentions direction, navigate there
        last_input = bundle.get('last_input') or ''
        last_input = last_input.lower()
        directions = ['north', 'south', 'east', 'west']
        direction = None
        for d in directions:
            if d in last_input:
                direction = d
                break
        if not direction:
            direction = 'north'  # default
        distance = 10.0  # default
        confidence = 0.7
        return {
            'type': 'navigate',
            'direction': direction,
            'distance': distance,
            'confidence': confidence,
            'source': 'PFC',
        }

    def process(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        hippocampus = bundle.get('hippocampus', {})
        matches = hippocampus.get('matches', [])  # Similar memories
        inputs_this_step = bundle.get('inputs_this_step') or []
        
        # Get VentralStriatum analysis
        vs_analysis = bundle.get('ventral_striatum', {})
        top_thought = vs_analysis.get('top_thought', {})
        scored_thoughts = vs_analysis.get('scored_thoughts', [])
        
        # Get simulation phase
        simulation_phase = bundle.get('simulation_phase', 'conversation')

        speak_action = self._build_speak_action(bundle)
        reflect_action = self._build_reflect_action(bundle)
        navigate_action = self._build_navigate_action(bundle)
        candidate_actions: List[Dict[str, Any]] = [speak_action, reflect_action, navigate_action]

        # Adjust confidence of speak action with similarity context
        if matches:
            top_sim = matches[0].get('similarity', 0.0)
            speak_action['confidence'] = round(min(1.0, speak_action['confidence'] + 0.1 * top_sim), 3)

        # Incorporate VS scoring into action selection
        if scored_thoughts:
            # Boost confidence of actions based on VS scores
            top_score = top_thought.get('overall_score', 0.5)
            tags = top_thought.get('tags', [])
            
            # If high reward/motivation, boost speak confidence
            if 'rewarding' in tags or 'motivating' in tags or top_score > 0.7:
                speak_action['confidence'] = round(min(1.0, speak_action['confidence'] + 0.2), 3)
            
            # If high safety/stability, boost reflect confidence
            if 'safe' in tags or 'stable' in tags:
                reflect_action['confidence'] = round(min(1.0, reflect_action['confidence'] + 0.15), 3)
            
            # If high urgency/attention, boost navigate confidence
            if 'urgent' in tags or 'attention' in tags:
                navigate_action['confidence'] = round(min(1.0, navigate_action['confidence'] + 0.15), 3)

        # Phase-specific adjustments
        if simulation_phase == 'mind_wander':
            # During mind wandering, strongly prefer reflection on memories
            reflect_action['confidence'] = round(min(1.0, reflect_action['confidence'] + 0.3), 3)
            # Reduce speak confidence during mind wandering
            speak_action['confidence'] = round(max(0.1, speak_action['confidence'] - 0.2), 3)

        # Selection policy:
        # - If there is new external input this step -> prioritize speak.
        # - If input mentions direction, prioritize navigate.
        # - Else prefer reflect (internal cognition) if current thought exists.
        if inputs_this_step:
            if any(d in ((bundle.get('last_input') or '').lower()) for d in ['north', 'south', 'east', 'west']):
                selected = navigate_action
                rationale = 'Input mentions direction; navigating.'
            else:
                selected = speak_action
                rationale = 'External input present; generating outward speech.'
        else:
            selected = reflect_action
            rationale = 'No new external input; engaging internal reflective cycle.'

        bundle['pfc'] = {
            'selected_action': selected,
            'candidate_actions': candidate_actions,
            'rationale': rationale,
            'vs_influence': {
                'top_score': top_thought.get('overall_score', 0.0) if top_thought else 0.0,
                'dominant_tags': top_thought.get('tags', [])[:3] if top_thought else []
            }
        }
        bundle['cognitive_actions'] = candidate_actions
        return bundle
