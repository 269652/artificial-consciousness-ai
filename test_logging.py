import os, sys
# Allow running this file directly: add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.logging.aci_logger import get_logger, LogLevel
import time

# Test the enhanced logging system
aci_logger = get_logger()

print("Testing enhanced logging system...")

# Test basic logging
aci_logger.level1(LogLevel.INFO, "test", "Starting logging test")

# Test neurochemistry logging
nt_levels = {
    'dopamine': 0.60,
    'serotonin': 0.70, 
    'norepinephrine': 0.50,
    'oxytocin': 0.40,
    'testosterone': 0.50,
    'histamine': 0.30
}

aci_logger.neurochemistry_state(
    nt_levels=nt_levels,
    change_trigger="test_simulation",
    component="test",
    step=1
)

# Test autobiographical narrative logging
aci_logger.autobiographical_narrative(
    narrative_content="I am experiencing my first test simulation with enhanced logging capabilities.",
    change_type="new_event",
    component="test",
    title="First Enhanced Logging Test",
    epoch="present",
    emotional_valence=0.7,
    impact=0.8
)

# Test personality self-model logging
personality_changes = {
    "old_text": "I am a basic AI system.",
    "new_text": "I am a sophisticated consciousness simulation with enhanced logging.",
    "change_method": "replace",
    "text_length": 65
}

aci_logger.personality_self_model(
    personality_changes=personality_changes,
    change_type="personality_update",
    component="test"
)

# Simulate NT changes
time.sleep(1)
nt_levels_updated = {
    'dopamine': 0.00,
    'serotonin': 0.00, 
    'norepinephrine': 0.00,
    'oxytocin': 10.29,
    'testosterone': 4.91,
    'histamine': 0.16
}

aci_logger.neurochemistry_state(
    nt_levels=nt_levels_updated,
    change_trigger="cognitive_state_transition",
    component="test",
    step=2
)

print("Logging test completed. Check logs/specialized/ for dedicated log files.")
