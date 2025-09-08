import os, sys
# Allow running this file directly: add project root (parent of 'src') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

from src.modules.DMN import DefaultModeNetwork
from src.modules.IdentityMemory import IdentityMemory
from src.logging.aci_logger import get_logger, LogLevel
from src.memory.persistent_memory import get_memory_manager
from src.memory.knowledge_extractor import get_knowledge_extractor
import time
import random
import requests
import uuid

# Initialize logging and memory systems
aci_logger = get_logger()

# Try to initialize memory manager, fall back to None if database unavailable
try:
    memory_manager = get_memory_manager()
    knowledge_extractor = get_knowledge_extractor()
    print("Database connections established")
except Exception as e:
    print(f"Database unavailable, running without persistent memory: {e}")
    memory_manager = None
    knowledge_extractor = None

# Perplexity API key from env
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY not set")

# Personalities
PERSONALITIES = {
    "Plato": "You are Plato, the ancient Greek philosopher. Speak in a wise, philosophical manner, discussing ideas, ethics, and the ideal forms.",
    "Einstein": "You are Albert Einstein, the physicist. Speak intelligently about science, relativity, and the universe.",
    "Hitler": "You are Adolf Hitler, the historical figure. Speak in a charismatic, ideological manner about politics and power. (Note: This is for simulation purposes only.)"
}

class SeedExperience:
    def __init__(self):
        aci_logger.level1(LogLevel.INFO, "seed_experience", "Initializing SeedExperience simulation")

        self.dmn = DefaultModeNetwork()
        self.identity_memory = IdentityMemory()
        self.step_count = 0
        self.world_time = 6.0  # Start at 6 AM
        self.current_personality = None
        self.conversation_turns = 0
        self.current_person_id = None
        self.session_id = f"seed_experience_{int(time.time())}"

        # Load existing memories to demonstrate persistence
        self._load_existing_memories()

        aci_logger.level1(LogLevel.INFO, "seed_experience", "SeedExperience initialized",
                         session_id=self.session_id)

    def _load_existing_memories(self):
        """Load and display existing memories to show persistence"""
        try:
            # Load recent episodic memories
            episodes = memory_manager.load_episodic_memories(limit=5)
            if episodes:
                aci_logger.level1("INFO", "seed_experience", "Loaded existing episodic memories",
                                 count=len(episodes))
                print(f"📚 Loaded {len(episodes)} existing episodic memories")

            # Load recent narrative memories
            narratives = memory_manager.load_narrative_memories(limit=3)
            if narratives:
                aci_logger.level1("INFO", "seed_experience", "Loaded existing narrative memories",
                                 count=len(narratives))
                print(f"📖 Loaded {len(narratives)} existing narrative memories")

            # Load knowledge graph
            nodes, edges = memory_manager.load_knowledge_graph()
            if nodes or edges:
                aci_logger.level1("INFO", "seed_experience", "Loaded existing knowledge graph",
                                 nodes=len(nodes), edges=len(edges))
                print(f"🕸️  Loaded knowledge graph with {len(nodes)} nodes and {len(edges)} edges")

        except Exception as e:
            aci_logger.error(f"Failed to load existing memories: {e}",
                           component="seed_experience")

    def simulate_world_change(self):
        # Simulate time passing, sun rising, sounds changing
        self.world_time += 0.1  # 10 minutes per step
        if self.world_time > 24:
            self.world_time -= 24
        time_of_day = "dawn" if self.world_time < 6 else "morning" if self.world_time < 12 else "afternoon" if self.world_time < 18 else "evening"
        sounds = "birds chirping" if time_of_day == "dawn" else "traffic" if time_of_day == "morning" else "wind" if time_of_day == "afternoon" else "crickets"
        return f"Time: {self.world_time:.1f} ({time_of_day}), Sounds: {sounds}"

    def generate_personality_response(self, personality, user_input):
        prompt = f"{PERSONALITIES[personality]}\nUser: {user_input}\n{personality}:"
        response = self.call_perplexity_api(prompt)
        return response

    def call_perplexity_api(self, prompt):
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "I'm sorry, I couldn't respond right now."

    def run_simulation(self, steps=100):
        aci_logger.level1("INFO", "seed_experience", "Starting simulation",
                         steps=steps, session_id=self.session_id)

        print("🧠 Starting ACI SeedExperience simulation with persistent memory...")
        print("📊 All activities will be logged and memories will persist across runs")
        print("=" * 60)

        conversation_steps = 8  # Increased to allow 4-7 speak actions
        mind_wander_steps = 7   # Increased to 7 steps
        current_phase = "conversation"
        phase_counter = 0
        personality_index = 0
        personalities_list = list(PERSONALITIES.keys())

        for step in range(steps):
            self.step_count += 1
            world_desc = self.simulate_world_change()

            aci_logger.level2("INFO", "seed_experience", "Simulation step started",
                             step=self.step_count, phase=current_phase)

            # Fake sensory input for Associative Cortex
            fake_sensory = {
                'timestamp': self.step_count,
                'world_time': self.world_time,
                'day_count': 1,
                'environmental': {
                    'time_of_day': world_desc.split('(')[1].split(')')[0],
                    'weather': {'condition': 'clear', 'intensity': 0.5, 'temperature': 20},
                    'terrain': {'type': 'urban', 'elevation': 0},
                    'atmosphere': {'humidity': 50, 'air_quality': 80},
                    'nearby_features': [{'type': 'bench', 'distance': 0}],
                    'acoustic_environment': [{'description': world_desc.split('Sounds: ')[1]}]
                },
                'spatial': {
                    'position': [100, 100],
                    'heading': 0.0,
                    'elevation': 0
                },
                'visual': {
                    'rgb_camera': None,  # Placeholder
                    'resolution': [400, 300],
                    'ambient_light': 1.0
                },
                'proximity': [],
                'proximity_detailed': []
            }

            # Phase control
            if current_phase == "conversation":
                if phase_counter == 0:
                    # Add a person to the world
                    personality_name = personalities_list[personality_index % len(personalities_list)]
                    self.current_person_id = self.dmn.world.add_person(personality_name)
                    self.current_personality = personality_name
                    self.conversation_turns = 0

                    aci_logger.world_event(f"A person resembling {self.current_personality} sits down on the bench",
                                          component="seed_experience")
                    print(f"[WORLD] A person resembling {self.current_personality} sits down on the bench next to you.")

                phase_counter += 1
                if phase_counter >= conversation_steps:
                    # Make person leave
                    goodbye = f"{self.current_personality} stands up and says goodbye."
                    self.dmn.world.update_person_action(self.current_person_id, 'speaking', goodbye)

                    aci_logger.world_event(f"{self.current_personality} says goodbye and leaves",
                                          component="seed_experience")
                    print(f"[WORLD] {goodbye}")

                    # Move person away
                    target_pos = (self.dmn.world.agent_pos[0] + 50, self.dmn.world.agent_pos[1] + 50)
                    self.dmn.world.move_person_toward(self.current_person_id, target_pos)

                    self.current_personality = None
                    self.current_person_id = None
                    self.conversation_turns = 0
                    current_phase = "mind_wander"
                    phase_counter = 0
                    personality_index += 1

                    aci_logger.level1("INFO", "seed_experience", "Phase transition: conversation -> mind_wander")
                    print("[PHASE] Switching to mind wandering...")

            elif current_phase == "mind_wander":
                phase_counter += 1
                if phase_counter >= mind_wander_steps:
                    # Switch back to conversation
                    current_phase = "conversation"
                    phase_counter = 0

                    aci_logger.level1("INFO", "seed_experience", "Phase transition: mind_wander -> conversation")
                    print("[PHASE] Switching to conversation...")

            # Handle conversation if personality is present
            if self.current_personality and self.conversation_turns < 6:  # Increased to allow 4-7 speak actions
                if self.conversation_turns == 0:
                    # Start conversation
                    start_prompt = "Hello, I'm waiting for the bus. What brings you here?"
                    self.dmn.world.update_person_action(self.current_person_id, 'speaking', start_prompt)

                    aci_logger.world_event(f"Conversation started with {self.current_personality}",
                                          component="seed_experience")
                    print(f"[AI] {start_prompt}")
                else:
                    # AI's response from previous step
                    last_response = self.dmn.last_bundle.get('current_thought', {}).get('text', '') if self.dmn.last_bundle else ''
                    if last_response:
                        personality_response = self.generate_personality_response(self.current_personality, last_response)
                        self.dmn.world.update_person_action(self.current_person_id, 'speaking', personality_response)

                        aci_logger.world_event(f"{self.current_personality} responds in conversation",
                                              component="seed_experience", response_length=len(personality_response))
                        print(f"[{self.current_personality}] {personality_response}")
                self.conversation_turns += 1

            # Update person positions and actions
            if self.current_person_id:
                distance = self.dmn.world.get_person_distance(self.current_person_id)
                if distance > 20 and not self.dmn.world.persons[0]['is_leaving']:  # If person is far and not leaving
                    # Move person closer for conversation
                    target_pos = (self.dmn.world.agent_pos[0] + 5, self.dmn.world.agent_pos[1] + 5)
                    self.dmn.world.move_person_toward(self.current_person_id, target_pos)
                elif self.dmn.world.persons[0]['is_leaving']:
                    # Continue moving away
                    target_pos = (self.dmn.world.agent_pos[0] + 50, self.dmn.world.agent_pos[1] + 50)
                    self.dmn.world.move_person_toward(self.current_person_id, target_pos)

                    # Remove person if far enough
                    if distance > 30:
                        self.dmn.world.remove_person(self.current_person_id)
                        self.current_person_id = None

            # Occasionally add internal thought (more frequent during mind wandering)
            thought_chance = 0.4 if current_phase == "mind_wander" else 0.2
            if random.random() < thought_chance:
                internal_thoughts = [
                    "I wonder when that bus will come.",
                    "This bench is uncomfortable.",
                    "The weather is nice today.",
                    "I should have brought a book.",
                    "Time seems to pass slowly when waiting.",
                    "What if the bus never comes?",
                    "I feel so alone here.",
                    "The world around me is changing.",
                    "My thoughts drift like clouds.",
                    "Is this all there is to existence?"
                ]
                thought = random.choice(internal_thoughts)
                self.dmn.add_input(thought)

                aci_logger.thought(thought, component="seed_experience")
                print(f"[INTERNAL] {thought}")

            # Update proximity_detailed with current person information
            if self.current_person_id:
                nearby_persons = self.dmn.world.get_nearby_persons(50)
                fake_sensory['proximity_detailed'] = nearby_persons

            # Step DMN
            result = self.dmn.step()

            # Inject fake sensory data for AssociativeCortex
            result['sensory_data'] = fake_sensory

            # Add current phase information to bundle for PFC decision making
            result['simulation_phase'] = current_phase

            # Log key info with enhanced logging
            print(f"[STEP {self.step_count}] {world_desc}")
            print(f"[INPUT] {result.get('last_input', 'None')}")

            current_thought = result.get('current_thought', {}).get('text', '')
            if current_thought:
                print(f"[THOUGHT] {current_thought}")

            selected_action = result.get('pfc', {}).get('selected_action', {})
            if selected_action:
                print(f"[ACTION] {selected_action}")

            executed = result.get('executed_actions', [])
            if executed:
                for ex in executed:
                    print(f"[EXECUTED] {ex}")

            # Log to our advanced logging system
            aci_logger.level2("INFO", "seed_experience", "Step completed",
                             step=self.step_count, phase=current_phase,
                             has_thought=bool(current_thought),
                             has_action=bool(selected_action))

            print("-" * 50)

            # End conversation after turns
            if self.conversation_turns >= 6:
                self.conversation_turns = 0

            time.sleep(0.5)  # Slow down for readability

        # Final knowledge extraction and summary
        try:
            knowledge_result = knowledge_extractor.extract_and_store_knowledge()
            aci_logger.level1("INFO", "seed_experience", "Final knowledge extraction completed",
                             **knowledge_result)
        except Exception as e:
            aci_logger.error(f"Final knowledge extraction failed: {e}",
                           component="seed_experience")

        aci_logger.level1("INFO", "seed_experience", "Simulation completed",
                         total_steps=self.step_count, session_id=self.session_id)
        print("🎉 Simulation complete. All memories have been persisted to database.")
        print("📊 Check the logs/ directory for detailed activity logs")
        print("🗄️  Memories will persist across future runs")

if __name__ == "__main__":
    se = SeedExperience()
    se.run_simulation(30)  # Run for 30 steps to see multiple phase cycles
