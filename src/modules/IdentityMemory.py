import uuid
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime

class PersonIdentity:
    """Represents a person's identity with traits and narrative"""
    def __init__(self, person_id: str, name: str, initial_traits: Dict[str, float] = None):
        self.person_id = person_id
        self.name = name
        self.traits = initial_traits or {}
        self.narrative = []
        self.encounter_count = 0
        self.last_seen = datetime.now()
        self.first_encounter = datetime.now()
        self.conversation_history = []

    def add_conversation(self, conversation_text: str):
        """Add a conversation snippet to build narrative"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'text': conversation_text
        })
        self.encounter_count += 1
        self.last_seen = datetime.now()

    def update_traits(self, new_traits: Dict[str, float]):
        """Update personality traits based on interactions"""
        for trait, value in new_traits.items():
            if trait in self.traits:
                # Weighted average with existing trait
                self.traits[trait] = (self.traits[trait] * self.encounter_count + value) / (self.encounter_count + 1)
            else:
                self.traits[trait] = value

    def add_narrative_element(self, element: str):
        """Add an element to the person's narrative"""
        self.narrative.append({
            'timestamp': datetime.now(),
            'element': element
        })

    def get_summary(self) -> str:
        """Get a summary of the person's identity"""
        traits_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.traits.items()])
        narrative_str = "; ".join([elem['element'] for elem in self.narrative[-3:]])  # Last 3 elements
        return f"{self.name} - Traits: {traits_str}. Recent narrative: {narrative_str}"

class IdentityMemory:
    """Memory system for storing and retrieving person identities"""

    def __init__(self, storage_path: str = None):
        self.identities: Dict[str, PersonIdentity] = {}
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "..", "..", "data", "identities.json")
        self._load_identities()

    def _load_identities(self):
        """Load identities from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for person_id, identity_data in data.items():
                        identity = PersonIdentity(
                            person_id=identity_data['person_id'],
                            name=identity_data['name'],
                            initial_traits=identity_data.get('traits', {})
                        )
                        identity.encounter_count = identity_data.get('encounter_count', 0)
                        identity.narrative = identity_data.get('narrative', [])
                        identity.conversation_history = identity_data.get('conversation_history', [])
                        if 'first_encounter' in identity_data:
                            identity.first_encounter = datetime.fromisoformat(identity_data['first_encounter'])
                        if 'last_seen' in identity_data:
                            identity.last_seen = datetime.fromisoformat(identity_data['last_seen'])
                        self.identities[person_id] = identity
            except Exception as e:
                print(f"[IdentityMemory] Failed to load identities: {e}")

    def _save_identities(self):
        """Save identities to storage"""
        try:
            data = {}
            for person_id, identity in self.identities.items():
                data[person_id] = {
                    'person_id': identity.person_id,
                    'name': identity.name,
                    'traits': identity.traits,
                    'narrative': identity.narrative,
                    'encounter_count': identity.encounter_count,
                    'conversation_history': identity.conversation_history,
                    'first_encounter': identity.first_encounter.isoformat(),
                    'last_seen': identity.last_seen.isoformat()
                }
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[IdentityMemory] Failed to save identities: {e}")

    def get_or_create_identity(self, person_id: str, name: str) -> PersonIdentity:
        """Get existing identity or create new one"""
        if person_id not in self.identities:
            self.identities[person_id] = PersonIdentity(person_id, name)
            self._save_identities()
        return self.identities[person_id]

    def update_from_conversation(self, person_id: str, conversation_text: str):
        """Update identity based on conversation content"""
        if person_id in self.identities:
            identity = self.identities[person_id]
            identity.add_conversation(conversation_text)

            # Extract traits from conversation (simple keyword-based analysis)
            traits = self._extract_traits_from_text(conversation_text)
            if traits:
                identity.update_traits(traits)

            # Add narrative elements
            narrative_element = self._extract_narrative_from_text(conversation_text, identity.name)
            if narrative_element:
                identity.add_narrative_element(narrative_element)

            self._save_identities()

    def _extract_traits_from_text(self, text: str) -> Dict[str, float]:
        """Simple trait extraction from conversation text"""
        traits = {}
        text_lower = text.lower()

        # Philosophical trait
        if any(word in text_lower for word in ['philosophy', 'wisdom', 'truth', 'forms', 'ideal']):
            traits['philosophical'] = 0.8

        # Scientific trait
        if any(word in text_lower for word in ['science', 'physics', 'theory', 'experiment']):
            traits['scientific'] = 0.8

        # Authoritative trait
        if any(word in text_lower for word in ['power', 'control', 'authority', 'leadership']):
            traits['authoritative'] = 0.7

        # Friendly trait
        if any(word in text_lower for word in ['friend', 'hello', 'nice', 'pleasant']):
            traits['friendly'] = 0.6

        return traits

    def _extract_narrative_from_text(self, text: str, name: str) -> str:
        """Extract narrative elements from conversation"""
        text_lower = text.lower()

        if 'philosophy' in text_lower or 'wisdom' in text_lower:
            return f"{name} engages in philosophical discussions"
        elif 'science' in text_lower or 'physics' in text_lower:
            return f"{name} discusses scientific concepts"
        elif 'power' in text_lower or 'politics' in text_lower:
            return f"{name} talks about power and politics"
        elif 'hello' in text_lower or 'greet' in text_lower:
            return f"{name} is sociable and initiates conversations"

        return ""

    def get_nearby_persons(self, current_pos: Tuple[float, float], max_distance: float = 20.0) -> List[Dict[str, Any]]:
        """Get information about nearby persons (to be integrated with world state)"""
        # This will be populated by the world simulation
        return []

    def recognize_person(self, visual_features: Dict[str, Any]) -> Optional[str]:
        """Recognize a person from visual features (placeholder for future implementation)"""
        # For now, return None - persons are identified by their unique IDs
        return None
