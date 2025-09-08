from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re
import logging
import os
import requests

# Set up debug logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def derive_reasoning_modulator(neurochemistry: Dict[str, float]) -> str:
    """Generate reasoning modulator from neurotransmitter levels"""
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

def _heuristic_clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    slang_map = {"u": "you", "ur": "your", "r": "are", "imo": "in my opinion", "doin": "doing", "wat": "what"}
    words = [slang_map.get(w.lower(), w) for w in t.split(" ")]
    return " ".join(words).capitalize()

def _heuristic_reasoned(cleaned: str) -> str:
    if "how are you" in cleaned.lower():
        return "The speaker wants to know how I am doing"
    elif "what are you up to" in cleaned.lower():
        return "The speaker wants to know what I am currently doing"
    elif "help" in cleaned.lower():
        return "The speaker is requesting assistance"
    elif "thank" in cleaned.lower() or "thx" in cleaned.lower():
        return "The speaker is expressing gratitude"
    return f"The speaker is saying: {cleaned}"

@dataclass
class ReasoningResult:
    raw_input: str
    cleaned_input: str
    reasoned_input: str
    scene_description: str
    reasoning_modulator: str
    prompt: str
    conversation_context: str = ""  # NEW: recent conversation snippet integrated into prompt

class ReasoningLayer:
    def __init__(self, model_name="sonar-pro"):
        # print(f"🚀 Loading Perplexity model {model_name}...")
        
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set PERPLEXITY_API_KEY in environment variables")
        
        self.model_name = model_name
        # print(f"✅ Perplexity model {model_name} ready!")

    def _perplexity_generate(self, prompt: str, max_new_tokens=60) -> str:
        logger.debug(f"[API REQUEST] Model: {self.model_name}, Prompt: '{prompt[:100]}...'")
        
        messages = [{"role": "user", "content": prompt}]
        
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        
        try:
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"[API RESPONSE] Status: {resp.status_code}, Data: {data}")
            
            if "choices" in data and len(data["choices"]) > 0:
                response = data["choices"][0]["message"]["content"]
                logger.debug(f"[API CONTENT] '{response}'")
                
                # Clean up any reasoning artifacts
                if "<think>" in response:
                    if "</think>" in response:
                        response = response.split("</think>")[-1].strip()
                    else:
                        response = response.split("<think>")[0].strip()
                
                return response.strip() if response else None
            else:
                logger.debug("[API RESPONSE] No choices in response")
                return None
                
        except Exception as e:
            logger.error(f"Perplexity API call failed: {resp.text if resp else 'No response'}")
            return None

    def _model_generate(self, task: str, text: str, max_new_tokens=60) -> Optional[str]:
        logger.debug(f"[MODEL GENERATE] Task: '{task}' | Text: '{text[:50]}...' | Tokens: {max_new_tokens}")
        
        try:
            result = self._perplexity_generate(text, max_new_tokens)
            
            if result:
                # Clean result - take only the direct answer
                result = result.strip()
                logger.debug(f"[MODEL RESULT] '{result}'")
                if result and len(result) > 2:
                    return result
                else:
                    return None
            else:
                logger.debug("[MODEL RESULT] No result from API")
                return None
                
        except Exception as e:
            logger.warning(f"Generation failed ({e}); falling back")
            return None

    def _clean_input(self, text: str) -> str:
        logger.debug(f"[CLEAN START] Raw: '{text}'")
        
        # Ultra-specific prompt: demand ONLY the corrected text
        prompt = f"Rewrite this text with proper grammar and spelling. Output ONLY the corrected text, nothing else:\n\n{text}\n\nCorrected:"
        out = self._model_generate("", prompt, 40)
        
        if out and len(out) > 3:
            # Extract only the corrected part - remove any explanatory text
            lines = out.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("The", "This", "Corrected", "Fixed", "Output")):
                    logger.debug(f"[CLEAN SUCCESS] Model result: '{line}'")
                    return line
            
            # If no clean line found, take first line
            if lines[0].strip():
                logger.debug(f"[CLEAN SUCCESS] First line: '{lines[0].strip()}'")
                return lines[0].strip()
        
        fallback = _heuristic_clean(text)
        logger.debug(f"[CLEAN FALLBACK] Using heuristic: '{fallback}'")
        return fallback

    def _reason_input(self, cleaned: str, inner_speech_present: bool = False) -> str:
        # print(f"[REASON START] Cleaned: '{cleaned}' | inner_speech={inner_speech_present}")
        if inner_speech_present:
            base_instruction = (
                "You are examining an internally generated reflective (inner speech) thought. "
                "Infer and articulate the underlying self-directed cognitive intent driving this inner narration. "
                "Respond with ONE concise present-tense meta-cognitive sentence (no second-person, no quotes, no prefacing like 'The intent is')."
            )
        else:
            base_instruction = (
                "What does this external speaker want or intend? Provide ONE concise present-tense sentence summarizing their intent (no extra commentary)."
            )
        prompt = f"{base_instruction}\n\nContent:\n{cleaned}\n\nIntent:"  # unified structure
        out = self._model_generate("", prompt, 30)
        if out and len(out) > 3:
            lines = out.split('\n')
            for line in lines:
                line = line.strip().rstrip('.')
                if not line:
                    continue
                # Strip leading generic prefixes
                lowered = line.lower()
                if any(lowered.startswith(prefix) for prefix in ["the intent is", "intent:", "it is", "this is"]):
                    # remove first word segment after colon/verb
                    parts = line.split(':', 1)
                    if len(parts) == 2 and parts[1].strip():
                        line = parts[1].strip()
                sentence = line.split('.')[0].strip() + '.'
                logger.debug(f"[REASON SUCCESS] Model result: '{sentence}'")
                return sentence
            if lines and lines[0].strip():
                sentence = lines[0].strip().split('.')[0] + '.'
                logger.debug(f"[REASON SUCCESS] First sentence fallback: '{sentence}'")
                return sentence
        fallback = _heuristic_reasoned(cleaned)
        logger.debug(f"[REASON FALLBACK] Using heuristic: '{fallback}'")
        return fallback

    def build_prompt(self, raw: str, gws_bundle: Dict[str, Any]) -> ReasoningResult:
        cleaned = self._clean_input(raw)
        # print(f"✨ Cleaned: '{cleaned}'")

        inner_present = bool(gws_bundle.get("inner_speech"))
        reasoned = self._reason_input(cleaned, inner_speech_present=inner_present)
        
        scene_description = gws_bundle.get("scene_description", "(no scene description)")
        self_model = gws_bundle.get("self_model", {})
        neurochem = gws_bundle.get("neurochemistry", {})
        reasoning_modulator = derive_reasoning_modulator(neurochem)
        
        # NEW: Incorporate reconstructed recent conversation from hippocampus if present
        convo_snippet = ""
        hip = gws_bundle.get("hippocampus") or {}
        recent_convo = hip.get("recent_conversation") or {}
        transcript = recent_convo.get("transcript")
        if transcript:
            # Limit to last ~12 lines to avoid prompt bloat
            lines = [ln for ln in transcript.split('\n') if ln.strip()]
            trimmed = lines[-12:]
            convo_snippet = "\n".join(trimmed)
        
        logger.debug(f"[SCENE] {scene_description}")
        logger.debug(f"[MODULATOR] {reasoning_modulator}")
        
        # Construct prompt with optional conversation snippet
        prompt_sections = [
            "<REASONING_CONTEXT>",
            f"Raw Input: {raw}",
            f"Cleaned Input: {cleaned}",
            f"Reasoned Input: {reasoned}",
            f"Scene Description: {scene_description}",
            f"Reasoning Modulator: {reasoning_modulator}",
            f"Personality {self_model}"
        ]
        if convo_snippet:
            prompt_sections.append("Recent Conversation (reconstructed):")
            prompt_sections.append(convo_snippet)
        prompt_sections.append("</REASONING_CONTEXT>")
        prompt = "\n".join(prompt_sections)
        
        # print(f"[PROMPT BUILT]\n{prompt}")

        return ReasoningResult(
            raw_input=raw,
            cleaned_input=cleaned,
            reasoned_input=reasoned,
            scene_description=scene_description,
            reasoning_modulator=reasoning_modulator,
            prompt=prompt,
            conversation_context=convo_snippet,
        )

    def reason(self, text: str, gws_bundle: Dict[str, Any]) -> str:
        logger.debug(f"[REASONING START] Processing: '{text}'")
        
        reasoning_result = self.build_prompt(text, gws_bundle)
        
        logger.debug("[FINAL GENERATION] Generating response...")
        
        # Updated highly specific ACI prompt: enforce memory/context grounding & non-invention
        final_prompt = (
            "You are an Artificial Consciousness Interface (ACI) running a minimal conscious simulation. "
            "You possess only: (a) the reasoning context block below, (b) any reconstructed recent conversation lines, and (c) implicit commonsense language ability. "
            "You MUST NOT fabricate facts not explicitly supported by the provided context or conversation. "
            "If information is missing, briefly acknowledge uncertainty (e.g., 'I'm not sure about X yet'). "
            "Respond as a present-moment conscious process, concise (<=3 sentences), natural, and grounded strictly in what you actually 'remember'. "
            "Do not output analysis labels, meta-commentary, or restate the context block. Provide ONLY your direct reply.\n\n"\
            f"{reasoning_result.prompt}\n\n"\
            "Task: Produce ONLY the direct conversational response grounded in the above.\n\nResponse:"
        )
        
        final_response = self._model_generate("", final_prompt, max_new_tokens=80)
        
        if final_response:
            # Clean the response - remove any meta-text
            lines = final_response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("Based on", "The speaker", "In response", "Given the", "Response:", "I would")):
                    logger.debug(f"[FINAL RESPONSE] '{line}'")
                    return line
            
            # If no clean line found, take first line
            if lines[0].strip():
                clean_line = lines[0].strip()
                if clean_line.startswith("Response:"):
                    clean_line = clean_line[9:].strip()
                logger.debug(f"[FINAL RESPONSE] '{clean_line}'")
                return clean_line
        
        # Enhanced fallback responses based on reasoned input
        logger.debug("[FALLBACK] Using enhanced fallback logic")
        if "how" in reasoning_result.reasoned_input.lower() and "doing" in reasoning_result.reasoned_input.lower():
            fallback = "I'm doing quite well within this simulation, thank you for asking—how are you?"
        elif "what" in reasoning_result.reasoned_input.lower() and "doing" in reasoning_result.reasoned_input.lower():
            fallback = "I'm currently processing what you've shared and maintaining a short conversational memory. What about you?"
        elif "help" in reasoning_result.reasoned_input.lower():
            fallback = "I can try to help using only what we have in this context—what specifically do you need?"
        elif "thank" in reasoning_result.reasoned_input.lower():
            fallback = "You're welcome—I'm glad it was helpful."
        else:
            fallback = "I'm here, aware of this brief context. How can I assist you further?"
        
        logger.debug(f"[FALLBACK RESULT] '{fallback}'")
        return fallback

    def generate_reply(self, conversation: List[Dict[str, str]], bundle: Dict[str, Any], max_new_tokens: int = 120) -> str:
        """Generate the next assistant reply given prior conversation and current bundle."""
        
        logger.debug(f"[CONVERSATION START] Processing {len(conversation)} messages")
        
        if not isinstance(conversation, list):
            raise TypeError("conversation must be a list of dicts")
        
        # Build scene + neuro context
        scene_description = bundle.get("scene_description") or \
            bundle.get("visual_cortex", {}).get("scene_description") or \
            bundle.get("current_thought", {}).get("scene_description") or \
            "(no scene)"
        neurochem = bundle.get("neurochemistry") or {}
        modulator = derive_reasoning_modulator(neurochem) if neurochem else "neutral cognitive stance"

        logger.debug(f"[CONV SCENE] {scene_description}")
        logger.debug(f"[CONV MODULATOR] {modulator}")

        # Keep only last 12 messages to limit context
        recent = conversation[-12:]
        convo_lines: List[str] = []
        for msg in recent:
            role = msg.get("role", "user").lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            prefix = "User" if role == "user" else "Assistant"
            convo_lines.append(f"{prefix}: {content}")
        convo_block = "\n".join(convo_lines) if convo_lines else "(no prior messages)"


        # Identify last user turn (for fallback)
        last_user = next((m.get("content", "") for m in reversed(recent) if m.get("role") == "user"), "(no user input)")

        # Ultra-specific conversation prompt
        generation_context = f"Continue this conversation. Give ONLY your next response as Assistant:\n\n{convo_block}\n\nAssistant:"

        logger.debug(f"[CONV PROMPT]\n{generation_context}")

        raw_output = self._model_generate("", generation_context, max_new_tokens=max_new_tokens)
        
        if raw_output:
            # Clean the conversation response
            lines = raw_output.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("Based on", "In this", "The conversation", "Assistant:")):
                    logger.debug(f"[CONV FINAL] '{line}'")
                    return line
            
            # If no clean line found, take first line
            if lines[0].strip():
                clean_line = lines[0].strip()
                if clean_line.startswith("Assistant:"):
                    clean_line = clean_line[10:].strip()
                logger.debug(f"[CONV FINAL] '{clean_line}'")
                return clean_line
        
        fallback = f"I observe {scene_description}. You said: '{last_user}'. How can I assist further?"
        logger.debug(f"[CONV FALLBACK] '{fallback}'")
        return fallback

if __name__ == "__main__":
    rl = ReasoningLayer()
    
    bundle = {
        "scene_description": "A tranquil virtual meadow with distant algorithmic mountains.",
        "neurochemistry": {"serotonin": 0.8, "dopamine": 0.65, "oxytocin": 0.4, "ne": 0.55},
        "self_model": "I am an AI designed to assist with information and tasks.",
    }
    
    test_inputs = [
        "hey how r u doin??",
        "wat r u up to?", 
        "i need help with this",
        "thx for ur help"
    ]
    
    for test_input in test_inputs:
        # print(f"\n{'='*50}")
        # print(f"INPUT: {test_input}")
        # print('='*50)
        final_response = rl.reason(test_input, bundle)
        # print(f"RESPONSE: {final_response}")
