from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modules.DMN import DefaultModeNetwork
from src.modules.ReasoningLayer import ReasoningLayer

logger = logging.getLogger(__name__)

def derive_reasoning_modulator(neurochemistry: Dict[str, float]) -> str:
    serotonin = neurochemistry.get('serotonin', 0.5)
    dopamine = neurochemistry.get('dopamine', 0.5)
    oxytocin = neurochemistry.get('oxytocin', 0.5)
    ne = neurochemistry.get('ne', 0.5)
    
    if serotonin > 0.7:
        mood = "optimistic and emotionally stable"
    elif serotonin < 0.3:
        mood = "pessimistic with emotional volatility"
    else:
        mood = "emotionally balanced"
    
    if dopamine > 0.7:
        drive = "highly motivated and reward-seeking"
    elif dopamine < 0.3:
        drive = "low motivation and reduced goal pursuit"
    else:
        drive = "moderate motivation levels"
    
    if oxytocin > 0.7:
        social = "highly empathetic and trusting, socially oriented"
    elif oxytocin < 0.3:
        social = "individualistic and socially cautious"
    else:
        social = "socially balanced"
    
    if ne > 0.7:
        alertness = "hypervigilant with rapid, stress-influenced thinking"
    elif ne < 0.3:
        alertness = "relaxed with slower cognitive processing"
    else:
        alertness = "alert and cognitively focused"
    
    return f"Neurochemical Profile: {mood}, {drive}, {social}, {alertness} [S:{serotonin:.2f} D:{dopamine:.2f} O:{oxytocin:.2f} NE:{ne:.2f}]"

if __name__ == "__main__":
    dmn = DefaultModeNetwork()
    
    # Use the factory function to create ReasoningLayer with proper configuration
    try:
        from src.config.reasoning_config import create_reasoning_layer
        reasoner = create_reasoning_layer()
    except ImportError:
        # Fallback to default Perplexity
        reasoner = ReasoningLayer()

    conversation = []  # list of {'role': 'user'|'assistant', 'content': str}

    initial_user_input = "Hello, describe what you perceive and introduce yourself."
    dmn.add_input(initial_user_input)
    conversation.append({"role": "user", "content": initial_user_input})
    print(f"[USER] {initial_user_input}")

    steps = 7
    for step in range(steps):
        bundle = dmn.step()

        thought_text = bundle.get("current_thought", {}).get("text")
        if thought_text:
            print(f"[THOUGHT] {thought_text}")

        executed = bundle.get("executed_actions", [])
        spoken_output = None
        for exec_item in reversed(executed):
            res = exec_item.get("result", {})
            if res.get("audible"):
                spoken_output = res.get("output")
                break

        if spoken_output:
            print(f"[ACI SPEAK] {spoken_output}")
            # Treat ACI speech as latest user-style message driving the assistant reply
            conversation.append({"role": "user", "content": spoken_output})
            # Generate assistant reply based on conversation & current bundle
            assistant_reply = reasoner.generate_reply(conversation, bundle, max_new_tokens=96)
            print(f"[ASSISTANT] {assistant_reply}")
            conversation.append({"role": "assistant", "content": assistant_reply})
            # Feed assistant reply back as next DMN input
            if step < steps - 1:
                dmn.add_input(assistant_reply)
        else:
            print("[ACI SPEAK] (no audible output)")

    print("\nFinal conversation transcript:")
    for turn in conversation:
        print(f"{turn['role'].upper()}: {turn['content']}")

    print("\nFinal executed actions last step:")
    for a in executed:
        print(a)

