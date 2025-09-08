import numpy as np
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import threading
import time

class Agent:
    """
    Intelligent agent that processes multi-modal sensory input streams
    and provides interfaces for consciousness research frameworks.
    """
    
    def __init__(self, agent_id: str, sensory_buffer_size: int = 1000):
        """
        Initialize agent with sensory processing capabilities
        
        Args:
            agent_id: Unique identifier for this agent
            sensory_buffer_size: Size of sensory input buffer for temporal processing
        """
        self.agent_id = agent_id
        self.world = None
        
        # Sensory input streams with temporal buffers
        self.sensory_streams = {
            'visual': deque(maxlen=sensory_buffer_size),
            'spatial': deque(maxlen=sensory_buffer_size), 
            'proximity': deque(maxlen=sensory_buffer_size),
            'proprioceptive': deque(maxlen=sensory_buffer_size),
            'environmental': deque(maxlen=sensory_buffer_size),
        }
        
        # Real-time sensory data (most recent)
        self.current_sensory_data = {}
        
        # Callbacks for consciousness framework integration
        self.sensory_callbacks = []
        self.decision_callbacks = []
        
        # Agent state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.internal_state = {
            'arousal': 0.5,
            'attention_focus': None,
            'memory_consolidation': False,
        }
        
        # Action output
        self.current_action = np.array([0.0, 0.0])  # [acceleration, steering]
        
    def set_world(self, world):
        """Connect agent to the simulated world"""
        self.world = world
        
    def update_sensory_input(self, sensory_data: Dict[str, Any], timestamp: int):
        """
        Update agent's sensory input streams with new data
        
        Args:
            sensory_data: Comprehensive sensory data from world
            timestamp: Current simulation timestamp
        """
        # Store current sensory state
        self.current_sensory_data = sensory_data.copy()
        self.current_sensory_data['timestamp'] = timestamp
        
        # Update temporal buffers
        for stream_type in self.sensory_streams.keys():
            if stream_type in sensory_data:
                self.sensory_streams[stream_type].append({
                    'timestamp': timestamp,
                    'data': sensory_data[stream_type]
                })
        
        # Update agent's physical state
        if 'spatial' in sensory_data:
            spatial = sensory_data['spatial']
            if 'position' in spatial:
                self.position = spatial['position']
            if 'heading' in spatial:
                self.orientation = spatial['heading']  
            if 'velocity' in spatial:
                self.velocity = spatial['velocity']
                
        # Trigger sensory processing callbacks
        self._trigger_sensory_callbacks(sensory_data, timestamp)
        
    def get_visual_stream(self, history_length: int = 1) -> List[Dict]:
        """Get visual sensory stream data"""
        return list(self.sensory_streams['visual'])[-history_length:]
        
    def get_spatial_stream(self, history_length: int = 1) -> List[Dict]:
        """Get spatial/kinematic sensory stream data"""
        return list(self.sensory_streams['spatial'])[-history_length:]
        
    def get_proximity_stream(self, history_length: int = 1) -> List[Dict]:
        """Get proximity sensor stream data"""
        return list(self.sensory_streams['proximity'])[-history_length:]
        
    def get_proprioceptive_stream(self, history_length: int = 1) -> List[Dict]:
        """Get proprioceptive sensory stream data"""
        return list(self.sensory_streams['proprioceptive'])[-history_length:]
        
    def get_environmental_stream(self, history_length: int = 1) -> List[Dict]:
        """Get environmental context stream data"""
        return list(self.sensory_streams['environmental'])[-history_length:]
        
    def get_current_sensory_state(self) -> Dict[str, Any]:
        """Get the most recent comprehensive sensory state"""
        return self.current_sensory_data.copy()
        
    def get_multimodal_sensory_fusion(self, history_length: int = 5) -> Dict[str, Any]:
        """
        Get fused multi-modal sensory data for consciousness processing
        
        Args:
            history_length: Number of recent timesteps to include
            
        Returns:
            Fused sensory representation suitable for cortex processing
        """
        fused_data = {
            'current': self.current_sensory_data,
            'temporal_context': {},
            'sensory_summary': {},
        }
        
        # Collect temporal context
        for modality in self.sensory_streams.keys():
            recent_data = list(self.sensory_streams[modality])[-history_length:]
            fused_data['temporal_context'][modality] = recent_data
            
        # Generate sensory summaries
        fused_data['sensory_summary'] = self._generate_sensory_summary()
        
        return fused_data
        
    def _generate_sensory_summary(self) -> Dict[str, Any]:
        """Generate high-level summary of current sensory state"""
        summary = {
            'scene_complexity': 0.0,
            'motion_state': 'stationary',
            'environmental_safety': 'safe',
            'attention_targets': [],
        }
        
        # Analyze visual complexity
        if self.current_sensory_data.get('visual', {}).get('rgb_camera') is not None:
            visual_data = self.current_sensory_data['visual']['rgb_camera']
            if isinstance(visual_data, np.ndarray):
                summary['scene_complexity'] = np.std(visual_data) / 255.0
                
        # Analyze motion state
        if 'spatial' in self.current_sensory_data:
            speed = self.current_sensory_data['spatial'].get('speed', 0)
            if speed > 2.0:
                summary['motion_state'] = 'moving_fast'
            elif speed > 0.1:
                summary['motion_state'] = 'moving_slow'
                
        # Analyze environmental safety
        if 'environmental' in self.current_sensory_data:
            collision_risk = self.current_sensory_data['environmental']['traffic_state'].get('collision_risk', False)
            summary['environmental_safety'] = 'danger' if collision_risk else 'safe'
            
        return summary
        
    def register_sensory_callback(self, callback: Callable):
        """Register callback for real-time sensory processing"""
        self.sensory_callbacks.append(callback)
        
    def register_decision_callback(self, callback: Callable):  
        """Register callback for decision/action processing"""
        self.decision_callbacks.append(callback)
        
    def _trigger_sensory_callbacks(self, sensory_data: Dict, timestamp: int):
        """Trigger registered sensory processing"""
