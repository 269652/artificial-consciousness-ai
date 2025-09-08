from __future__ import annotations
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BodyExecutor:
    """Body Execution Module.

    Executes body actions via BodyModel.
    """

    def __init__(self):
        self._counter = 0

    def _execute_navigate(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        direction = action.get('direction', 'north')
        distance = action.get('distance', 10.0)
        success = body_model.navigate(direction, distance)
        return {
            'effector': 'body_model',
            'action': 'navigate',
            'direction': direction,
            'distance': distance,
            'success': success
        }

    def _execute_push_object(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        object_id = action.get('object_id', 'unknown')
        force = action.get('force', 10.0)
        direction = action.get('direction', 'forward')
        success = body_model.push_object(object_id, force, direction)
        return {
            'effector': 'body_model',
            'action': 'push_object',
            'object_id': object_id,
            'force': force,
            'direction': direction,
            'success': success
        }

    def _execute_hit_object(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        object_id = action.get('object_id', 'unknown')
        force = action.get('force', 10.0)
        success = body_model.hit_object(object_id, force)
        return {
            'effector': 'body_model',
            'action': 'hit_object',
            'object_id': object_id,
            'force': force,
            'success': success
        }

    def _execute_talk(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        message = action.get('message', '')
        target = action.get('target')
        success = body_model.talk(message, target)
        return {
            'effector': 'body_model',
            'action': 'talk',
            'message': message,
            'target': target,
            'success': success
        }

    def _execute_jump(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        height = action.get('height', 1.0)
        success = body_model.jump(height)
        return {
            'effector': 'body_model',
            'action': 'jump',
            'height': height,
            'success': success
        }

    def _execute_wave(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        direction = action.get('direction', 'left')
        success = body_model.wave(direction)
        return {
            'effector': 'body_model',
            'action': 'wave',
            'direction': direction,
            'success': success
        }

    def _execute_yell(self, action: Dict[str, Any], body_model) -> Dict[str, Any]:
        message = action.get('message', '')
        success = body_model.yell(message)
        return {
            'effector': 'body_model',
            'action': 'yell',
            'message': message,
            'success': success
        }

    def process(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        actions: List[Dict[str, Any]] = bundle.get('cognitive_actions') or []
        executed: List[Dict[str, Any]] = bundle.get('executed_actions', [])
        body_model = bundle.get('body_model')

        if not body_model:
            logger.warning("No body_model in bundle")
            return bundle

        for act in actions:
            a_type = act.get('type')
            result = None
            if a_type == 'navigate':
                result = self._execute_navigate(act, body_model)
            elif a_type == 'push_object':
                result = self._execute_push_object(act, body_model)
            elif a_type == 'hit_object':
                result = self._execute_hit_object(act, body_model)
            elif a_type == 'talk':
                result = self._execute_talk(act, body_model)
            elif a_type == 'jump':
                result = self._execute_jump(act, body_model)
            elif a_type == 'wave':
                result = self._execute_wave(act, body_model)
            elif a_type == 'yell':
                result = self._execute_yell(act, body_model)
            else:
                logger.debug(f"BodyExecutor skipping unsupported action type: {a_type}")
                continue
            self._counter += 1
            executed.append({
                'id': f"exec_{self._counter}",
                'action': act,
                'result': result
            })

        bundle['executed_actions'] = executed
        return bundle
