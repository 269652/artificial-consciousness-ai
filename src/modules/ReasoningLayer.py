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
    def __init__(self, model_name="sonar-pro", provider="perplexity"):
        """
        Initialize ReasoningLayer with support for multiple API providers.
        
        Args:
            model_name: Model name to use (e.g., "sonar-pro" for Perplexity, "deepseek-chat" for DeepSeek)
            provider: API provider ("perplexity" or "deepseek")
        """
        self.provider = provider.lower()
        self.model_name = model_name
        
        # Initialize API credentials based on provider
        if self.provider == "perplexity":
            self.api_key = os.getenv("PERPLEXITY_API_KEY")
            if not self.api_key:
                raise RuntimeError("Set PERPLEXITY_API_KEY in environment variables")
            self.api_url = "https://api.perplexity.ai/chat/completions"
            logger.info(f"✅ Perplexity ReasoningLayer initialized with model {model_name}")
        elif self.provider == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise RuntimeError("Set DEEPSEEK_API_KEY in environment variables")
            self.api_url = "https://api.deepseek.com/chat/completions"
            logger.info(f"✅ DeepSeek ReasoningLayer initialized with model {model_name}")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'perplexity' or 'deepseek'")

    def _api_generate(self, prompt: str, max_new_tokens=60) -> str:
        """Unified API generation method supporting both Perplexity and DeepSeek."""
        logger.debug(f"[API REQUEST] Provider: {self.provider}, Model: {self.model_name}, Prompt: '{prompt[:100]}...'")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare request parameters
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # DeepSeek-specific adjustments
        if self.provider == "deepseek":
            # DeepSeek uses "max_tokens" but may have different parameter preferences
            request_data.update({
                "stream": False,  # Ensure non-streaming response
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            })
        
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_data
        )
        
        try:
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"[API RESPONSE] Provider: {self.provider}, Status: {resp.status_code}")
            
            if "choices" in data and len(data["choices"]) > 0:
                response = data["choices"][0]["message"]["content"]
                logger.debug(f"[API CONTENT] '{response}'")
                
                # Clean up any reasoning artifacts (common to both providers)
                if "<think>" in response:
                    if "</think>" in response:
                        response = response.split("</think>")[-1].strip()
                    else:
                        response = response.split("<think>")[0].strip()
                
                return response.strip() if response else None
            else:
                logger.debug(f"[API RESPONSE] No choices in response from {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"{self.provider.title()} API call failed: {resp.text if resp else 'No response'}")
            return None

    # Legacy method for backward compatibility
    def _perplexity_generate(self, prompt: str, max_new_tokens=60) -> str:
        """Legacy method - use _api_generate for better provider support."""
        return self._api_generate(prompt, max_new_tokens)

    def _model_generate(self, task: str, text: str, max_new_tokens=60) -> Optional[str]:
        logger.debug(f"[MODEL GENERATE] Provider: {self.provider}, Task: '{task}' | Text: '{text[:50]}...' | Tokens: {max_new_tokens}")
        
        try:
            result = self._api_generate(text, max_new_tokens)
            
            if result:
                # Clean result - take only the direct answer
                result = result.strip()
                logger.debug(f"[MODEL RESULT] '{result}'")
                if result and len(result) > 2:
                    return result
                else:
                    return None
            else:
                logger.debug(f"[MODEL RESULT] No result from {self.provider} API")
                return None
                
        except Exception as e:
            logger.warning(f"Generation failed with {self.provider} ({e}); falling back")
            return None

    def _process_input_combined(self, text: str, inner_speech_present: bool = False) -> tuple[str, str]:
        """Optimized method that combines cleaning and reasoning into a single API call."""
        logger.debug(f"[COMBINED PROCESSING START] Raw: '{text}' | inner_speech={inner_speech_present}")
        
        # Determine reasoning instruction based on inner speech context
        if inner_speech_present:
            reasoning_instruction = (
                "Infer and articulate the underlying self-directed cognitive intent driving this inner narration. "
                "Provide ONE concise present-tense meta-cognitive sentence (no second-person, no quotes, no prefacing like 'The intent is')."
            )
        else:
            reasoning_instruction = (
                "What does this external speaker want or intend? Provide ONE concise present-tense sentence summarizing their intent (no extra commentary)."
            )
        
        # Combined prompt for both cleaning and reasoning
        prompt = f"""Perform two tasks on this text:

1. CLEANING: Rewrite the text with proper grammar and spelling. Output only the corrected text.
2. REASONING: {reasoning_instruction}

Text to process:
{text}

Respond in this exact format:
CLEANED: [corrected text here]
INTENT: [intent analysis here]"""
        
        out = self._model_generate("", prompt, 80)
        
        if out and len(out) > 10:
            # Parse the structured response
            lines = out.strip().split('\n')
            cleaned_text = ""
            reasoned_text = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("CLEANED:"):
                    cleaned_text = line.replace("CLEANED:", "").strip()
                elif line.startswith("INTENT:"):
                    reasoned_text = line.replace("INTENT:", "").strip()
            
            # Fallback parsing if structured format not followed
            if not cleaned_text or not reasoned_text:
                logger.debug("[COMBINED] Structured format not followed, using fallback parsing")
                text_parts = out.split('\n', 1)
                if len(text_parts) >= 2:
                    cleaned_text = text_parts[0].strip()
                    reasoned_text = text_parts[1].strip()
                else:
                    cleaned_text = out.strip()
                    reasoned_text = out.strip()
            
            # Clean up the results
            if cleaned_text:
                # Remove any prefixes from cleaned text
                for prefix in ["CLEANED:", "Corrected:", "Fixed:", "Output:"]:
                    if cleaned_text.startswith(prefix):
                        cleaned_text = cleaned_text.replace(prefix, "").strip()
                
            if reasoned_text:
                # Clean up reasoning result
                reasoned_text = reasoned_text.rstrip('.')
                # Strip leading generic prefixes
                lowered = reasoned_text.lower()
                if any(lowered.startswith(prefix) for prefix in ["the intent is", "intent:", "it is", "this is"]):
                    parts = reasoned_text.split(':', 1)
                    if len(parts) == 2 and parts[1].strip():
                        reasoned_text = parts[1].strip()
                reasoned_text = reasoned_text.split('.')[0].strip() + '.'
            
            if cleaned_text and reasoned_text:
                logger.debug(f"[COMBINED SUCCESS] Cleaned: '{cleaned_text}' | Intent: '{reasoned_text}'")
                return cleaned_text, reasoned_text
        
        # Fallback to heuristic methods if API call fails
        logger.debug("[COMBINED FALLBACK] Using heuristic methods")
        cleaned_fallback = _heuristic_clean(text)
        reasoned_fallback = _heuristic_reasoned(cleaned_fallback)
        return cleaned_fallback, reasoned_fallback

    def _clean_input(self, text: str) -> str:
        """Legacy method - use _process_input_combined for better performance."""
        cleaned, _ = self._process_input_combined(text, False)
        return cleaned

    def _reason_input(self, cleaned: str, inner_speech_present: bool = False) -> str:
        """Legacy method - use _process_input_combined for better performance."""
        _, reasoned = self._process_input_combined(cleaned, inner_speech_present)
        return reasoned

    def build_prompt(self, raw: str, gws_bundle: Dict[str, Any]) -> ReasoningResult:
        # OPTIMIZED: Use combined processing to reduce API calls by 50%
        inner_present = bool(gws_bundle.get("inner_speech"))
        cleaned, reasoned = self._process_input_combined(raw, inner_present)
        
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
