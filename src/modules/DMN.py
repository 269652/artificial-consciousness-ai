import os, sys
# Allow running this file directly: add project root (parent of 'src') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    dotenv_path = os.path.join(project_root, ".env")
    env_txt_path = os.path.join(project_root, ".env.txt")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    elif os.path.exists(env_txt_path):
        load_dotenv(dotenv_path=env_txt_path)
except ImportError:
    pass  # python-dotenv not installed, skip

from src.simulations.World import SimpleWorld
from src.modules.VisualCortex import VisualCortex
from src.modules.ThoughtLayer import ThoughtLayer
from src.modules.Hippocampus import Hippocampus
from src.modules.PrefrontalCortex import PrefrontalCortex
from src.modules.MotorSpeechExecutor import MotorSpeechExecutor
from src.modules.InnerSpeech import InnerSpeech
from src.modules.AutobiographicalMemory import AutobiographicalMemory  # NEW
from src.modules.SelfModel import SelfModelLayer  # NEW
from src.modules.BodyModel import BodyModel
from src.modules.BodyExecutor import BodyExecutor
from src.memory.graph.controller import MemoryGraphController
from src.memory.graph.node import Node  # NEW import for memory logging
from src.memory.graph.Thought import Thought  # NEW import for embedding
from src.modules.IdentityMemory import IdentityMemory
from src.modules.VentralStriatum import VentralStriatum
from src.modules.NeuroChemistry import NeuroChemistry
from src.logging.aci_logger import get_logger, LogLevel

# Try to import optional dependencies
try:
    from src.memory.persistent_memory import get_memory_manager
    _PERSISTENT_MEMORY_AVAILABLE = True
except ImportError:
    _PERSISTENT_MEMORY_AVAILABLE = False
    def get_memory_manager():
        return None

try:
    from src.memory.knowledge_extractor import get_knowledge_extractor
    _KNOWLEDGE_EXTRACTOR_AVAILABLE = True
except ImportError:
    _KNOWLEDGE_EXTRACTOR_AVAILABLE = False
    def get_knowledge_extractor():
        return None
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import math
import time  # NEW import for timing steps

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
aci_logger = get_logger()

class DefaultModeNetwork:
    """
    DMN that steps world and passes bundle through sequential brain areas
    with persistent memory and advanced logging
    """
    def __init__(self, world_config=None, auto_log_memory: bool = True, relation_recompute_depth: int = 1):
        # Initialize logging and persistent memory first
        try:
            self.memory_manager = get_memory_manager()
        except Exception:
            self.memory_manager = None
            
        try:
            self.knowledge_extractor = get_knowledge_extractor()
        except Exception:
            self.knowledge_extractor = None

        self.world = SimpleWorld(**(world_config or {}))
        self.visual_cortex = VisualCortex()
        self.thought_layer = ThoughtLayer()
        self.hippocampus = Hippocampus()
        self.pfc = PrefrontalCortex()
        self.mse = MotorSpeechExecutor()
        self.inner_speech = InnerSpeech()
        # Core memory controller
        self.memory_graph_controller = MemoryGraphController()
        # Seed narrative + self model layers
        self.autobio = AutobiographicalMemory(self.memory_graph_controller, auto_seed=True, depth=1)
        self.self_model = SelfModelLayer(self.memory_graph_controller)
        # State
        self.current_step = 0
        # Maintain a rolling list of text inputs (raw user inputs)
        self._input_history: List[str] = []
        self._pending_inputs: List[str] = []  # inputs provided since last step
        # Config
        self.auto_log_memory = auto_log_memory
        self.relation_recompute_depth = relation_recompute_depth
        self.lastmemorynode = None
        self.last_bundle: Optional[Dict[str, Any]] = None
        # New: Floating memory node and thought
        self.floating_memory_node: Optional[Node] = None
        self.floating_thought: Optional[Dict[str, Any]] = None
        self.floating_similarity_threshold = 0.7  # Cosine similarity threshold for merging
        # New: Markov chains for action-response patterns
        self.markov_chains: Dict[str, Dict[str, int]] = {}  # e.g., {'Action: Greet': {'Response: Initiate': 1}}
        # Body model
        self.body_model = BodyModel()
        # Sync initial position
        self.body_model.position = tuple(self.world.agent_pos)
        self.body_executor = BodyExecutor()
        # Identity memory for person recognition
        self.identity_memory = IdentityMemory()
        # Ventral Striatum for thought scoring
        self.ventral_striatum = VentralStriatum()
        # NeuroChemistry for NT dynamics
        self.neurochemistry = NeuroChemistry()

        # Load persistent memory graph
        self._load_persistent_memory_graph()

        aci_logger.level1(LogLevel.INFO, "dmn", "Default Mode Network initialized with persistent memory")
 
    # --- New public API ---
    def seed_from_conversation(self, path: str, user_tag: str = "user_msg", assistant_tag: str = "assistant_msg") -> Dict[str, Any]:
        """Seed the shared memory graph controller from a conversation JSON file."""
        summary = self.memory_graph_controller.seed_from_conversation_json(path, tag_user=user_tag, tag_assistant=assistant_tag)
        return summary

    def add_input(self, text: str):
        """Add external textual input to be incorporated next step."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        stripped = text.strip()
        if stripped:
            self._pending_inputs.append(stripped)

    def get_input_history(self, n: Optional[int] = None) -> List[str]:
        """Return last n inputs (all if n None). Older -> newer."""
        if n is None or n >= len(self._input_history):
            return list(self._input_history)
        return self._input_history[-n:]

    def _autobio_summary(self) -> Dict[str, Any]:
        g = self.memory_graph_controller.graph
        nodes = [n for n in g.nodes.values() if 'autobio' in n.tags]
        return {
            'count': len(nodes),
            'titles': [n.context.get('title') for n in nodes if n.context.get('title')],
        }

    def _self_model_summary(self) -> Dict[str, Any]:
        return self.self_model.export_persona()

    def step(self, action=None):
        """
        Step world and process bundle through brain areas with persistent memory and logging
        """
        start_time = time.time()
        aci_logger.level2(LogLevel.INFO, "dmn", "Starting DMN step", step_number=self.current_step)

        # Default action
        if action is None:
            action = [1.0, np.random.uniform(-0.2, 0.2)]

        # Initial sensory input
        initial_sensory = self.world.step(action)
        aci_logger.level3(LogLevel.DEBUG, "dmn", "World step completed", sensory_keys=list(initial_sensory.keys()))

        # Consolidate inputs for this step
        inputs_this_step = self._pending_inputs
        if inputs_this_step:
            self._input_history.extend(inputs_this_step)
        last_input = inputs_this_step[-1] if inputs_this_step else None
        self._pending_inputs = []  # reset

        # Check for proximity-based communication from nearby persons
        proximity_inputs = self._get_proximity_inputs(initial_sensory)
        if proximity_inputs:
            inputs_this_step.extend(proximity_inputs)
            if not last_input:
                last_input = proximity_inputs[-1]
            self._input_history.extend(proximity_inputs)

        # Log input processing
        if last_input:
            aci_logger.level1(LogLevel.INFO, "dmn", "Processing input", input_text=last_input[:100])

        # Create initial bundle
        bundle: Dict[str, Any] = {
            'sensory_data': initial_sensory,
            'step': self.current_step,
            'action': action,
            'inputs_this_step': inputs_this_step,
            'last_input': last_input,
            'input_history_size': len(self._input_history),
            'memory_graph_controller': self.memory_graph_controller,
            'autobiographical_memory': self._autobio_summary(),
            'self_model': self._self_model_summary(),
            'body_model': self.body_model,
            'identity_memory': self.identity_memory,
            'neurochemistry': self.neurochemistry.get_current_state(),
        }

        # Pass through Visual Cortex (enriches with scene descriptions)
        bundle = self.visual_cortex.process(bundle)
        aci_logger.level3(LogLevel.DEBUG, "dmn", "Visual cortex processing completed")

        # If no external input, we pre-build a reflective action & inner speech seed
        if not last_input:
            # Ensure PFC reflect action logic has a placeholder selected_action for modulation
            # We'll run PFC after hippocampus; here we can optionally prime with previous bundle state
            # Generate provisional reflective inner speech (will be overwritten if PFC later selects speak)
            # Temporarily simulate reflect action meta for depth estimation
            provisional = {
                'pfc': {
                    'selected_action': {'type': 'reflect', 'content': bundle.get('last_input') or '', 'meta': {'depth_score': 0.5}}
                }
            }
            bundle.update(provisional)
            bundle = self.inner_speech.process(bundle)

        # Thought compilation layer (now aware of possible inner speech)
        bundle = self.thought_layer.process(bundle)
        current_thought = bundle.get('current_thought', {}).get('text', '')
        if current_thought:
            aci_logger.thought(current_thought, component="dmn")

        # Hippocampal retrieval BEFORE logging
        bundle = self.hippocampus.process(bundle)

        if (bundle.get("hippocampus") or {}).get("recent_conversation", {}).get("transcript"):
            bundle = self.thought_layer.process(bundle)

        # Ventral Striatum analysis of thought graph
        hippocampus_results = bundle.get('hippocampus', {})
        neurochemistry_state = bundle.get('neurochemistry', {})
        nt_levels = neurochemistry_state.get('nt_levels', {})

        vs_analysis = self.ventral_striatum.analyze_thought_graph(
            hippocampus_results,
            nt_levels
        )
        bundle['ventral_striatum'] = vs_analysis

        # Prefrontal cortex selects action (now with VS scoring)
        bundle = self.pfc.process(bundle)
        selected_action = bundle.get('pfc', {}).get('selected_action', {})
        if selected_action:
            aci_logger.action(selected_action.get('type', 'unknown'),
                            selected_action.get('content', ''),
                            component="dmn")

        # If PFC decided reflect and we did not already modulate with final action, run inner speech again
        if bundle.get('pfc', {}).get('selected_action', {}).get('type') == 'reflect':
            bundle = self.inner_speech.process(bundle)
            # Recompute thought embedding with inner speech influence
            bundle = self.thought_layer.process(bundle)

        # --- Automatic memory logging of current thought (moved AFTER PFC / inner speech) ---
        # Rationale: ensure episodic storage includes any PFC-selected action metadata in context
        if self.auto_log_memory:
            ct = bundle.get('current_thought') or {}
            emb = ct.get('embedding') or []
            text = ct.get('text') or None
            if emb and text:
                node_id = f"dmn_thought_{self.current_step}"
                if node_id not in self.memory_graph_controller.graph.nodes:
                    context = {
                        'text': text,
                        'step': self.current_step,
                        'source': 'DMN',
                        'last_input': bundle.get('last_input'),
                        'selected_action': bundle.get('pfc', {}).get('selected_action'),
                        'inner_speech': bundle.get('inner_speech'),
                        'autobio': bundle.get('autobiographical_memory'),
                        'self_model': bundle.get('self_model')
                    }
                    node = Node(node_id=node_id, embedding=emb, tags=['thought', 'conversation'], context=context)
                    if self.lastmemorynode:
                        self.memory_graph_controller.graph.edges.add_edge(node.id, self.lastmemorynode.id, 'conversation', weight=1)
                    self.lastmemorynode = node
                    self.memory_graph_controller.insert_node_with_relations(node, depth=self.relation_recompute_depth)
                    bundle['logged_memory_node_id'] = node_id
                    aci_logger.memory_operation("save", "memory_graph", component="dmn",
                                              node_id=node_id, step=self.current_step)
                else:
                    bundle['logged_memory_node_id'] = node_id

        # Motor speech executes
        bundle = self.mse.process(bundle)

        # Body executes
        bundle = self.body_executor.process(bundle)

        # Sync world position with body_model
        if hasattr(self.world, 'agent_pos'):
            self.world.agent_pos = np.array(self.body_model.position)

        # Step world with updated position
        sensory_data = self.world.step([0.0, 0.0])  # No movement, just update state

        # Update bundle with latest sensory
        bundle['sensory_data'] = sensory_data

        # Update floating thought and memory node
        ct = bundle.get('current_thought') or {}
        if ct.get('text'):
            # Add events to floating memory node
            if self.floating_memory_node:
                events = self.floating_memory_node.context.get('events', [])
                # Add input
                if last_input:
                    events.append({'type': 'input', 'content': last_input, 'step': self.current_step})
                # Add thought
                events.append({'type': 'thought', 'content': ct.get('text', ''), 'step': self.current_step})
                # Add action
                pfc_action = bundle.get('pfc', {}).get('selected_action', {})
                if pfc_action.get('content'):
                    events.append({'type': 'action', 'content': pfc_action.get('content', ''), 'step': self.current_step})
                    # Flag as response if previous was speak
                    if events and events[-2].get('type') == 'action' and 'speak' in events[-2]['content']:
                        events[-1]['causal_link'] = 'response'
                self.floating_memory_node.context['events'] = events

                # Consolidate events
                consolidated = self._consolidate_events(events)
                self.floating_memory_node.context['consolidated'] = consolidated
            # Check for significant context change
            context_changed = False
            if self.floating_thought:
                current_emb = self.floating_thought.get('embedding', [])
                new_emb = ct.get('embedding', [])
                if current_emb and new_emb:
                    sim = self._cosine(current_emb, new_emb)
                    if sim < 0.5 or pfc_action.get('type') == 'speak':
                        context_changed = True
            else:
                context_changed = True

            if context_changed and self.floating_memory_node:
                # Persist current floating memory node
                aci_logger.memory_operation("persist", "floating_memory", component="dmn",
                                          node_id=self.floating_memory_node.id)
                # Optionally, create a new floating memory node
                self.floating_memory_node = None

            # Update floating thought: merge or replace based on PFC decision or similarity
            if self.floating_thought is None:
                self.floating_thought = ct
            else:
                # Simple merge: append new text if similar, else replace
                current_emb = self.floating_thought.get('embedding', [])
                new_emb = ct.get('embedding', [])
                if current_emb and new_emb:
                    sim = self._cosine(current_emb, new_emb)
                    if sim > self.floating_similarity_threshold:
                        # Merge: combine texts
                        combined_text = f"{self.floating_thought['text']} {ct['text']}".strip()
                        self.floating_thought['text'] = combined_text[:500]  # Limit length
                        self.floating_thought['embedding'] = Thought.from_text(combined_text).embedding
                        aci_logger.level3(LogLevel.DEBUG, "dmn", "Merged floating thought", similarity=sim)
                    else:
                        # Replace if PFC decides or low similarity
                        pfc_action = bundle.get('pfc', {}).get('selected_action', {})
                        if pfc_action.get('type') == 'speak' or sim < 0.5:  # Low similarity or speak action
                            self.floating_thought = ct
                            aci_logger.level3(LogLevel.DEBUG, "dmn", "Replaced floating thought")
                else:
                    self.floating_thought = ct

            # Update floating memory node
            if self.floating_memory_node is None:
                # Create new floating memory node
                node_id = f"floating_memory_{self.current_step}"
                self.floating_memory_node = Node(
                    node_id=node_id,
                    embedding=self.floating_thought.get('embedding', []),
                    tags=['floating', 'memory'],
                    context={
                        'text': self.floating_thought.get('text', ''),
                        'start_step': self.current_step,
                        'last_update': self.current_step,
                        'layer': 'DMN',
                        'events': []  # Ordered list of events
                    }
                )
                self.memory_graph_controller.insert_node_with_relations(self.floating_memory_node, depth=self.relation_recompute_depth)
                aci_logger.memory_operation("create", "floating_memory", component="dmn", node_id=node_id)
            else:
                # Update existing floating memory node
                self.floating_memory_node.context['text'] = self.floating_thought.get('text', '')
                self.floating_memory_node.context['last_update'] = self.current_step
                self.floating_memory_node.embedding = self.floating_thought.get('embedding', [])
                aci_logger.level3(LogLevel.DEBUG, "dmn", "Updated floating memory node")

        # Update neurochemistry based on current cognitive state
        release_signals = self._generate_release_signals(bundle)
        self.neurochemistry.update(dt=1.0, release_signals=release_signals)

        # Update bundle with new neurochemistry state
        bundle['neurochemistry'] = self.neurochemistry.get_current_state()

        # Enhanced neurochemistry logging with state changes
        nt_levels = bundle['neurochemistry'].get('nt_levels', {})
        change_trigger = bundle.get('neurochemistry', {}).get('last_trigger', '')
        aci_logger.neurochemistry_state(
            nt_levels=nt_levels, 
            change_trigger=change_trigger,
            component="dmn",
            step=self.current_step,
            cycle_duration=time.time() - start_time
        )

        # Periodic knowledge extraction (every 10 steps)
        if self.current_step % 10 == 0 and self.knowledge_extractor:
            try:
                knowledge_result = self.knowledge_extractor.extract_and_store_knowledge()
                aci_logger.level2(LogLevel.INFO, "dmn", "Periodic knowledge extraction completed",
                                **knowledge_result)
            except Exception as e:
                aci_logger.error(f"Failed periodic knowledge extraction: {e}", component="dmn")

        # Periodic memory graph persistence (every 5 steps)
        if self.current_step % 5 == 0 and self.memory_manager:
            self._save_memory_graph_to_persistent()

        # Log final response
        final_response = bundle.get('current_thought', {}).get('text', '')
        if final_response:
            aci_logger.level1(LogLevel.INFO, "dmn", "Generated response", response=final_response[:100])

        # Performance metrics
        step_duration = time.time() - start_time
        aci_logger.performance_metric("dmn_step_duration", step_duration, "seconds", component="dmn")

        # Persist last bundle for next step reference
        self.last_bundle = bundle

        self.current_step += 1
        return bundle

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _abstract_event(self, event_type: str, content: str) -> str:
        """Simple abstraction for events."""
        if event_type == 'input':
            if 'how are you' in content.lower():
                return 'Input: Inquiry about state'
            elif 'what have you been up to' in content.lower():
                return 'Input: Inquiry about activities'
            return f'Input: {content[:50]}...'
        elif event_type == 'thought':
            if 'fine' in content.lower():
                return 'Thought: Positive state'
            return f'Thought: {content[:50]}...'
        elif event_type == 'action':
            if 'speak' in content.lower():
                return 'Action: Speak'
            return f'Action: {content[:50]}...'
        return f'{event_type}: {content[:50]}...'

    def _update_markov(self, prev_abstract: str, current_abstract: str):
        """Update Markov chains."""
        if prev_abstract not in self.markov_chains:
            self.markov_chains[prev_abstract] = {}
        if current_abstract not in self.markov_chains[prev_abstract]:
            self.markov_chains[prev_abstract][current_abstract] = 0
        self.markov_chains[prev_abstract][current_abstract] += 1

    def _consolidate_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate events into abstractions and update Markov chains."""
        abstractions = []
        prev_abstract = None
        for event in events:
            abstract = self._abstract_event(event['type'], event['content'])
            abstractions.append(abstract)
            if prev_abstract:
                self._update_markov(prev_abstract, abstract)
            prev_abstract = abstract
        consolidated_text = " ".join(abstractions)
        return {
            'abstractions': abstractions,
            'consolidated_text': consolidated_text,
            'markov_snapshot': dict(self.markov_chains)  # Snapshot of current chains
        }

    def _get_proximity_inputs(self, sensory_data: Dict[str, Any]) -> List[str]:
        """Get inputs from nearby persons within hearing range"""
        inputs = []
        proximity_detailed = sensory_data.get('proximity_detailed', [])
        
        for item in proximity_detailed:
            if item.get('type') == 'person':
                distance = item.get('distance', 0)
                action = item.get('action', '')
                speech_content = item.get('speech_content', '')
                person_id = item.get('person_id', '')
                name = item.get('name', 'Unknown Person')
                
                # Only hear speech within 15 meters
                if action == 'speaking' and distance <= 15 and speech_content:
                    inputs.append(f"{name}: {speech_content}")
                    
                    # Update identity memory with conversation
                    if person_id:
                        self.identity_memory.update_from_conversation(person_id, speech_content)
        
        return inputs

    def _generate_release_signals(self, bundle: Dict[str, Any]) -> Dict[str, float]:
        """Generate NT release signals based on current cognitive state"""
        
        signals = {}
        
        # Get current state information
        pfc_action = bundle.get('pfc', {}).get('selected_action', {})
        action_type = pfc_action.get('type', '')
        current_thought = bundle.get('current_thought', {})
        thought_text = current_thought.get('text', '').lower()
        
        vs_analysis = bundle.get('ventral_striatum', {})
        top_thought = vs_analysis.get('top_thought', {})
        overall_score = top_thought.get('overall_score', 0.5) if top_thought else 0.5
        
        # Base release signals
        signals = {
            'dopamine': 0.0,
            'serotonin': 0.0,
            'norepinephrine': 0.0,
            'oxytocin': 0.0,
            'testosterone': 0.0
        }
        
        # Dopamine release based on reward and exploration
        if action_type == 'speak' or 'curious' in thought_text or 'explore' in thought_text:
            signals['dopamine'] += 0.3
        if overall_score > 0.7:  # High scoring thought
            signals['dopamine'] += 0.2
        
        # Serotonin release based on safety and stability
        if 'safe' in thought_text or 'comfortable' in thought_text or action_type == 'reflect':
            signals['serotonin'] += 0.3
        if overall_score > 0.6:  # Moderately good thought
            signals['serotonin'] += 0.1
        
        # Norepinephrine release based on attention and urgency
        if 'urgent' in thought_text or 'important' in thought_text or 'focus' in thought_text:
            signals['norepinephrine'] += 0.4
        if action_type == 'speak':  # Speaking requires attention
            signals['norepinephrine'] += 0.2
        
        # Oxytocin release based on social interactions
        proximity_inputs = bundle.get('inputs_this_step', [])
        social_content = any('person:' in inp or 'social' in inp.lower() for inp in proximity_inputs)
        if social_content or 'friend' in thought_text or 'together' in thought_text:
            signals['oxytocin'] += 0.3
        
        # Testosterone release based on assertiveness and drive
        if 'drive' in thought_text or 'assertive' in thought_text or 'goal' in thought_text:
            signals['testosterone'] += 0.2
        
        # Add some baseline activity
        for nt in signals:
            signals[nt] = max(0.0, min(1.0, signals[nt] + 0.05))  # Clamp between 0 and 1
        
        return signals

    def _load_persistent_memory_graph(self):
        """Load memory graph from persistent storage"""
        try:
            nodes, edges = self.memory_manager.load_memory_graph()
            aci_logger.memory_operation("load", "memory_graph", component="dmn",
                                      nodes_loaded=len(nodes), edges_loaded=len(edges))

            # Add nodes to memory graph controller
            for node_data in nodes:
                node = Node(
                    node_id=node_data['id'],
                    embedding=node_data.get('embedding'),
                    tags=node_data.get('tags', []),
                    timestamp=node_data.get('timestamp'),
                    context=node_data.get('context', {}),
                    thought_chain=node_data.get('thought_chain', []),
                    neurochemistry=node_data.get('neurochemistry', {})
                )
                self.memory_graph_controller.graph.add_node(node)

            # Add edges to memory graph controller
            for edge_data in edges:
                self.memory_graph_controller.graph.add_edge(
                    edge_data['from_node_id'],
                    edge_data['to_node_id'],
                    edge_data['relation_type'],
                    edge_data.get('weight', 1.0)
                )

        except Exception as e:
            aci_logger.error(f"Failed to load persistent memory graph: {e}",
                           component="dmn")

    def _save_memory_graph_to_persistent(self):
        """Save current memory graph to persistent storage"""
        try:
            # Save nodes
            for node_id, node in self.memory_graph_controller.graph.nodes.items():
                self.memory_manager.save_memory_graph_node(
                    node_id=node_id,
                    embedding=node.embedding,
                    tags=node.tags,
                    timestamp=node.timestamp,
                    context=node.context,
                    thought_chain=node.thought_chain,
                    neurochemistry=node.neurochemistry
                )

            # Save edges
            for from_id, edges in self.memory_graph_controller.graph.edges.adj_list.items():
                for to_id, relation_type, weight in edges:
                    self.memory_manager.save_memory_graph_edge(
                        from_node_id=from_id,
                        to_node_id=to_id,
                        relation_type=relation_type,
                        weight=weight
                    )

            aci_logger.memory_operation("save", "memory_graph", component="dmn",
                                      nodes_saved=len(self.memory_graph_controller.graph.nodes),
                                      edges_saved=self.memory_graph_controller.graph.edges.size())

        except Exception as e:
            aci_logger.error(f"Failed to save memory graph to persistent storage: {e}",
                           component="dmn")
