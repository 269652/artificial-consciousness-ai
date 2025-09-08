import math
import random
from typing import Dict, Any
import os
import requests  # Added for API calls

class AssociativeCortex:
    def __init__(self, api_key=None, device=None, model_name="sonar-pro"):
        """Lazy-load Perplexity API; fallback to heuristic scene synthesis if unavailable."""
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model_name = model_name
        self.fallback = False
        if not self.api_key:
            print("[AssociativeCortex] Perplexity API key not provided; using fallback mode.")
            self.fallback = True
        # Removed T5-related code

    def generate_description(self, sensory_inputs: Dict[str, Any], prompt_prefix="describe scene:") -> str:
        # Check for persons in proximity
        persons_description = self._describe_nearby_persons(sensory_inputs)
        if persons_description:
            return persons_description
        
        if self.fallback:
            return self._fallback_description(sensory_inputs)
        prompt = f"{prompt_prefix} {self._create_structured_prompt(sensory_inputs)}"
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            }
            response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()
            if prompt_prefix.lower() in description.lower():
                description = description.replace(prompt_prefix, "").strip()
            return description
        except Exception as e:
            print(f"[AssociativeCortex] API call failed ({e}); using fallback mode.")
            self.fallback = True
            return self._fallback_description(sensory_inputs)

    def _fallback_description(self, sensory_inputs: Dict[str, Any]) -> str:
        env = sensory_inputs.get('environmental', {})
        terrain = env.get('terrain', {})
        weather = env.get('weather', {})
        feats = env.get('nearby_features', [])
        sounds = env.get('acoustic_environment', [])
        parts = []
        parts.append(f"Weather {weather.get('condition','clear')} temp {weather.get('temperature',20):.1f}C")
        parts.append(f"Terrain {terrain.get('type','unknown')} elev {terrain.get('elevation',0):.0f}")
        if feats:
            parts.append(f"Features: {', '.join(f.get('type','?') for f in feats[:3])}")
        if sounds:
            parts.append(f"Sounds: {', '.join(s.get('description','') for s in sounds[:2])}")
        return " | ".join(parts)

    def _create_structured_prompt(self, sensory_inputs: Dict[str, Any]) -> str:
        scenario_starters = [
            "I find myself in a mystical forest where",
            "I am exploring an ancient garden where",
            "I stand in a magical landscape where",
            "I wander through an enchanted realm where",
            "I discover a hidden valley where"
        ]
        base_scenario = random.choice(scenario_starters)
        natural_elements = []
        spatial = sensory_inputs.get('spatial', {})
        pos = spatial.get('position', [0, 0])
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            if pos[0] < 150:
                natural_elements.append("ancient stone formations rise to my west")
            elif pos[0] > 250:
                natural_elements.append("crystal pools shimmer to my east")
            else:
                natural_elements.append("rolling meadows stretch around me")
            if pos[1] < 100:
                natural_elements.append("mountain peaks tower in the distance")
            elif pos[1] > 200:
                natural_elements.append("a misty river flows nearby")
        proximity = sensory_inputs.get('proximity', {})
        if proximity:
            min_dist = min(v['distance'] if isinstance(v, dict) and 'distance' in v else v for v in proximity.values()) if proximity else 999
            if min_dist < 30:
                natural_elements.append("massive oak trees create a natural archway before me")
            elif min_dist < 60:
                natural_elements.append("flowering bushes dot the landscape ahead")
            else:
                natural_elements.append("an open clearing invites exploration")
        environmental = sensory_inputs.get('environmental', {})
        if environmental.get('weather', {}).get('condition'):
            cond = environmental['weather']['condition']
            natural_elements.append(f"the air carries a hint of {cond}")
        if natural_elements:
            description = f"{base_scenario} {', '.join(natural_elements[:3])}."
        else:
            description = f"{base_scenario} endless possibilities await discovery."
        return description

    def _heading_to_direction(self, heading_radians):
        """Convert heading in radians to cardinal direction"""
        
        # Normalize to 0-2π
        heading = heading_radians % (2 * math.pi)
        
        # Convert to degrees for easier calculation
        degrees = math.degrees(heading)
        
        # Map to cardinal directions
        if degrees < 45 or degrees >= 315:
            return "east"
        elif degrees < 135:
            return "north"
        elif degrees < 225:
            return "west"
        else:
            return "south"

    def _describe_nearby_persons(self, sensory_inputs: Dict[str, Any]) -> str:
        """Describe nearby persons if any are detected"""
        proximity_detailed = sensory_inputs.get('proximity_detailed', [])
        
        persons = []
        for item in proximity_detailed:
            if item.get('type') == 'person':
                person_id = item.get('person_id', 'unknown')
                name = item.get('name', 'Unknown Person')
                distance = item.get('distance', 0)
                action = item.get('action', 'standing')
                direction = item.get('direction', 'nearby')
                
                if distance <= 15:  # Only describe persons within hearing range
                    if action == 'leaving':
                        persons.append(f"{name} is leaving. {name} is {distance:.1f}m away")
                    elif action == 'approaching':
                        persons.append(f"{name} is approaching. {name} is {distance:.1f}m away")
                    elif action == 'speaking':
                        persons.append(f"{name} is speaking nearby")
                    else:
                        persons.append(f"{name} is {action} {distance:.1f}m away")
        
        if persons:
            return ". ".join(persons)
        
        return ""

    def describe_person_action(self, person_name: str, action: str, distance: float) -> str:
        """Generate description of a person's action"""
        if action == 'leaving':
            return f"{person_name} is leaving. {person_name} is {distance:.1f}m away"
        elif action == 'approaching':
            return f"{person_name} is approaching. {person_name} is {distance:.1f}m away"
        elif action == 'speaking':
            return f"{person_name} is speaking {distance:.1f}m away"
        else:
            return f"{person_name} is {action} {distance:.1f}m away"

# Test the refactored implementation
if __name__ == "__main__":
    # Test with sample sensory data (provide API key or it will fallback)
    ac = AssociativeCortex(api_key="your_api_key_here")  # Replace with actual key or set env var
    
    sample_sensory_data = {
        'spatial': {'position': [100.5, 200.3], 'heading': 1.57},
        'proximity': {'detector_0': 45.2, 'detector_1': 80.1},
        'visual': {'rgb_camera': [[1, 2, 3]] * 100},  # Simulate visual data
        'environmental': {'traffic_state': {'collision_risk': False}}
    }
    
    description = ac.generate_description(sample_sensory_data)
    print(f"Generated scene description: {description}")
