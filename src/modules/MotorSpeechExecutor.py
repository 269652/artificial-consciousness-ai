from __future__ import annotations
from typing import Dict, Any, List
import logging
import sys

logger = logging.getLogger(__name__)

class MotorSpeechExecutor:
    """Motor/Speech Execution Module (neuroanatomical analogy scaffold).

    Maps high-level PFC action plans to effectors:
      - 'speak' -> external speech output pathway (Broca's area -> articulators)
      - 'inner_speech' -> internal narration (no external output)
      - Future: writing, gesture, locomotion.

    Execution semantics:
      - Consumes bundle['pfc']['selected_action'] or iterates bundle['cognitive_actions'].
      - Appends executed actions with timestamps to bundle['executed_actions'].
    """

    def __init__(self, echo: bool = True):
        self.echo = echo
        self._counter = 0

    def _execute_speak(self, action: Dict[str, Any]) -> Dict[str, Any]:
        text = action.get('content', '')
        if self.echo:
            # print(f"[SPEAK] {text}")
            pass
        return {
            'effector': 'vocal_apparatus',
            'output': text,
            'audible': True
        }

    def _execute_inner(self, action: Dict[str, Any]) -> Dict[str, Any]:
        text = action.get('content', '')
        if self.echo:
            # print(f"[INNER SPEECH] {text}")
            pass
        return {
            'effector': 'inner_loop',
            'output': text,
            'audible': False
        }

    def process(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        actions: List[Dict[str, Any]] = bundle.get('cognitive_actions') or []
        executed: List[Dict[str, Any]] = bundle.get('executed_actions', [])

        for act in actions:
            a_type = act.get('type')
            if a_type == 'speak':
                result = self._execute_speak(act)
            elif a_type == 'inner_speech':
                result = self._execute_inner(act)
            else:
                # Unknown action type; skip or log
                logger.debug(f"MotorSpeechExecutor skipping unsupported action type: {a_type}")
                continue
            self._counter += 1
            executed.append({
                'id': f"exec_{self._counter}",
                'action': act,
                'result': result
            })

        bundle['executed_actions'] = executed
        return bundle
