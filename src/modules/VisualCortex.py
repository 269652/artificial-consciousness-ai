from src.modules.AssociativeCortex import AssociativeCortex

class VisualCortex:
    def __init__(self, model_name="sonar-pro"):
        self.associative_cortex = AssociativeCortex(model_name=model_name)
    
    def process(self, bundle):
        """Process bundle with enhanced world generation using rich environmental data"""
        sensory_data = bundle.get('sensory_data', {})
        
        # Extract rich environmental context
        environmental_context = self._build_environmental_context(sensory_data)
        
        # Use enhanced creative hallucination prompt with environmental details
        prefix = ("Transform this rich sensory data into a vivid fantasy world description. "
                  f"Environmental context: {environmental_context}. "
                  "Create an immersive scene that incorporates the time, weather, terrain, nearby obstacles, sounds, and atmosphere:")
        if getattr(self.associative_cortex, 'fallback', False):
            # Simpler description in fallback
            prefix = "describe scene:"
        scene_description = self.associative_cortex.generate_description(sensory_data, prompt_prefix=prefix)
        
        bundle['scene_description'] = scene_description
        bundle['environmental_context'] = environmental_context
        bundle['visual_cortex_processed'] = True
        
        return bundle
    
    def _build_environmental_context(self, sensory_data):
        """Build rich environmental context string from sensory data"""
        env = sensory_data.get('environmental', {})
        
        # Time and weather
        time_info = f"Time: {sensory_data.get('world_time', 0):.1f} ({env.get('time_of_day', 'unknown')})"
        
        weather = env.get('weather', {})
        weather_info = f"Weather: {weather.get('condition', 'clear')} ({weather.get('intensity', 0.5):.1f}) {weather.get('temperature', 20):.1f}C"
        
        # Terrain and environment
        terrain = env.get('terrain', {})
        terrain_info = f"Terrain: {terrain.get('type', 'unknown')} elev {terrain.get('elevation', 0):.0f}"
        
        # Atmosphere
        atmosphere = env.get('atmosphere', {})
        atm_info = f"Humidity {atmosphere.get('humidity', 50):.0f}% AirQ {atmosphere.get('air_quality', 100):.0f}"
        
        # Nearby features
        features = env.get('nearby_features', [])
        features_info = f"Features {len(features)}"
        
        # Sounds
        sounds = env.get('acoustic_environment', [])
        sound_desc = ', '.join(s.get('description','') for s in sounds[:2]) or 'quiet'
        
        return f"{time_info}. {weather_info}. {terrain_info}. {atm_info}. {features_info}. Sounds: {sound_desc}."
