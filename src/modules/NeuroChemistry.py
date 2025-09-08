from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
import random
import math

logger = logging.getLogger(__name__)

class NeuroChemistry:
    """NeuroChemistry module for simulating neurotransmitter dynamics.
    
    Implements simplified scalar dynamics for NT levels, binding, reuptake,
    and receptor interactions based on the ACI blueprint design.
    """
    
    def __init__(self):
        # NT state per area (simplified to single area for now)
        self.nt_levels = {
            'dopamine': 0.5,
            'serotonin': 0.6,  # Increased from 0.5 to 0.6
            'norepinephrine': 0.5,
            'oxytocin': 0.5,
            'testosterone': 0.5,
            'histamine': 0.3,
            'acetylcholine': 0.4,
            'glutamate': 0.6,
            'gaba': 0.4
        }
        
        # Receptor states
        self.receptors = {
            'dopamine': {
                'density': 100.0,
                'bound': 0.0,
                'k_on': 0.1,
                'k_off': 0.05,
                'kd': 0.5
            },
            'serotonin': {
                'density': 80.0,
                'bound': 0.0,
                'k_on': 0.08,
                'k_off': 0.04,
                'kd': 0.5
            },
            'norepinephrine': {
                'density': 60.0,
                'bound': 0.0,
                'k_on': 0.12,
                'k_off': 0.06,
                'kd': 0.5
            },
            'oxytocin': {
                'density': 40.0,
                'bound': 0.0,
                'k_on': 0.06,
                'k_off': 0.03,
                'kd': 0.5
            }
        }
        
        # Transporter states
        self.transporters = {
            'dopamine': {
                'density': 50.0,
                'k_reuptake': 0.1,
                'scaling': 1.0
            },
            'serotonin': {
                'density': 40.0,
                'k_reuptake': 0.08,
                'scaling': 1.0
            },
            'norepinephrine': {
                'density': 30.0,
                'k_reuptake': 0.12,
                'scaling': 1.0
            }
        }
        
        # Release pools for emitters
        self.release_pools = {
            'dopamine': 10.0,
            'serotonin': 8.0,
            'norepinephrine': 6.0,
            'oxytocin': 4.0,
            'testosterone': 3.0
        }
        
        # Diffusion/decay coefficients
        self.diffusion_coeffs = {
            'dopamine': 0.1,
            'serotonin': 0.08,
            'norepinephrine': 0.12,
            'oxytocin': 0.05,
            'testosterone': 0.03,
            'histamine': 0.15,
            'acetylcholine': 0.2,
            'glutamate': 0.25,
            'gaba': 0.2
        }
        
        # Homeostatic targets
        self.homeostatic_targets = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.4,
            'oxytocin': 0.3
        }
        
        # Learning rates for homeostasis
        self.homeostatic_learning_rate = 0.01
        self.transporter_learning_rate = 0.005
        
        # Track history for analysis
        self.history = []
        
    def update(self, dt: float = 1.0, release_signals: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Update NT dynamics for one time step"""
        
        release_signals = release_signals or {}
        
        # 1. Release from pools
        for nt, signal in release_signals.items():
            if nt in self.release_pools and nt in self.nt_levels:
                release_amount = signal * self.release_pools[nt] * dt
                self.nt_levels[nt] += release_amount
                self.release_pools[nt] = max(0, self.release_pools[nt] - release_amount)
        
        # 2. Binding/Unbinding for receptors
        for nt, receptor in self.receptors.items():
            if nt in self.nt_levels:
                nt_level = self.nt_levels[nt]
                
                # Mass-action binding
                free_receptors = receptor['density'] - receptor['bound']
                bind_rate = receptor['k_on'] * free_receptors * nt_level * dt
                unbind_rate = receptor['k_off'] * receptor['bound'] * dt
                
                receptor['bound'] = max(0, min(receptor['density'], 
                                             receptor['bound'] + bind_rate - unbind_rate))
                
                # Unbound NT goes back to pool
                unbound_amount = unbind_rate
                self.nt_levels[nt] += unbound_amount
        
        # 3. Reuptake by transporters
        for nt, transporter in self.transporters.items():
            if nt in self.nt_levels:
                reuptake_rate = (transporter['k_reuptake'] * 
                               transporter['scaling'] * 
                               transporter['density'] * 
                               self.nt_levels[nt] * dt)
                self.nt_levels[nt] = max(0, self.nt_levels[nt] - reuptake_rate)
        
        # 4. Diffusion/Decay
        for nt, level in self.nt_levels.items():
            decay = self.diffusion_coeffs.get(nt, 0.1) * level * dt
            self.nt_levels[nt] = max(0, level - decay)
        
        # 5. Homeostatic adjustments
        self._apply_homeostatic_adjustments(dt)
        
        # 6. Record state
        current_state = {
            'nt_levels': self.nt_levels.copy(),
            'receptor_occupancy': self.get_receptor_occupancy(),
            'release_pools': self.release_pools.copy(),
            'timestamp': len(self.history)
        }
        self.history.append(current_state)
        
        return current_state
    
    def get_receptor_occupancy(self) -> Dict[str, float]:
        """Get current receptor occupancy percentages"""
        occupancy = {}
        for nt, receptor in self.receptors.items():
            if receptor['density'] > 0:
                occupancy[nt] = receptor['bound'] / receptor['density']
            else:
                occupancy[nt] = 0.0
        return occupancy
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current neurochemical state"""
        return {
            'nt_levels': self.nt_levels.copy(),
            'receptor_occupancy': self.get_receptor_occupancy(),
            'release_pools': self.release_pools.copy(),
            'transporter_states': {nt: t.copy() for nt, t in self.transporters.items()},
            'receptor_states': {nt: r.copy() for nt, r in self.receptors.items()}
        }
    
    def modulate_release(self, modulation_signals: Dict[str, float]):
        """Apply modulation to release pools"""
        for nt, modulation in modulation_signals.items():
            if nt in self.release_pools:
                # Modulation can increase or decrease release pool
                self.release_pools[nt] *= (1.0 + modulation)
                self.release_pools[nt] = max(0, self.release_pools[nt])
    
    def _apply_homeostatic_adjustments(self, dt: float):
        """Apply homeostatic adjustments to maintain target levels"""
        for nt, target in self.homeostatic_targets.items():
            if nt in self.nt_levels:
                current = self.nt_levels[nt]
                error = target - current
                
                # Adjust receptor density
                if nt in self.receptors:
                    adjustment = self.homeostatic_learning_rate * error * dt
                    self.receptors[nt]['density'] = max(10.0, 
                                                      self.receptors[nt]['density'] + adjustment)
                
                # Adjust transporter density
                if nt in self.transporters:
                    adjustment = self.transporter_learning_rate * error * dt
                    self.transporters[nt]['density'] = max(5.0, 
                                                         self.transporters[nt]['density'] - adjustment)
    
    def get_effective_signals(self) -> Dict[str, float]:
        """Get effective NT signals based on receptor occupancy"""
        signals = {}
        occupancy = self.get_receptor_occupancy()
        
        for nt, occ in occupancy.items():
            # Apply allosteric modulation (simplified)
            allosteric_factor = 1.0
            if nt in self.receptors:
                # Simple allosteric effect based on other NT levels
                if nt == 'dopamine':
                    # DA modulated by serotonin
                    serotonin_occ = occupancy.get('serotonin', 0.5)
                    allosteric_factor = 1.0 + 0.2 * (serotonin_occ - 0.5)
                elif nt == 'serotonin':
                    # 5HT modulated by oxytocin
                    oxytocin_occ = occupancy.get('oxytocin', 0.3)
                    allosteric_factor = 1.0 + 0.3 * (oxytocin_occ - 0.3)
            
            signals[nt] = occ * allosteric_factor
        
        return signals
    
    def reset_to_baseline(self):
        """Reset all levels to baseline values"""
        for nt in self.nt_levels:
            self.nt_levels[nt] = 0.5
        
        for receptor in self.receptors.values():
            receptor['bound'] = 0.0
        
        for nt in self.release_pools:
            self.release_pools[nt] = 10.0  # Reset pools
