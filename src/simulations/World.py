import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from typing import Dict, Any, List, Tuple
import math
import random
from datetime import datetime, timedelta
from enum import Enum

class WeatherType(Enum):
    CLEAR = "clear"
    RAINY = "rainy"
    FOGGY = "foggy"
    STORMY = "stormy"
    SNOWY = "snowy"

class TimeOfDay(Enum):
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"

class ObstacleType(Enum):
    ROCK = "rock"
    TREE = "tree"
    BUILDING = "building"
    WATER = "water"
    CRYSTAL = "crystal"
    RUINS = "ruins"

class SimpleWorld:
    """Enhanced world simulation with rich environmental features"""
    
    def __init__(self, width=400, height=300):
        self.width = width
        self.height = height
        
        # Time and weather system
        self.world_time = 6.0  # Hour of day (0-24)
        self.day_count = 0
        self.weather = WeatherType.CLEAR
        self.weather_intensity = 0.5  # 0-1
        self.temperature = 20.0  # Celsius
        
        # Environmental features
        self.terrain_elevation = self._generate_terrain()
        self.vegetation_density = self._generate_vegetation()
        self.water_bodies = self._generate_water_bodies()
        
        # Enhanced obstacles
        self.obstacles = self._generate_rich_obstacles()
        self.interactive_objects = self._generate_interactive_objects()
        
        # Agent properties
        self.agent_pos = np.array([width//2, height//2], dtype=float)
        self.agent_angle = 0.0
        self.agent_health = 100.0
        self.agent_energy = 100.0
        self.step_count = 0
        
        # Persons in the world
        self.persons = []  # List of person dictionaries
        
        # Environmental effects
        self.wind_direction = random.uniform(0, 2 * math.pi)
        self.wind_speed = random.uniform(0, 10)
        self.ambient_light = 1.0
        
        # Create figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
    def _generate_terrain(self):
        """Generate elevation map using Perlin-like noise"""
        elevation = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                # Simple multi-octave noise
                elevation[i, j] = (
                    0.5 * math.sin(i * 0.02) * math.cos(j * 0.02) +
                    0.3 * math.sin(i * 0.05) * math.sin(j * 0.03) +
                    0.2 * math.sin(i * 0.1) * math.cos(j * 0.08)
                ) * 50 + 50
        return elevation
    
    def _generate_vegetation(self):
        """Generate vegetation density map"""
        vegetation = np.zeros((self.height, self.width))
        np.random.seed(123)
        for i in range(self.height):
            for j in range(self.width):
                # Vegetation depends on elevation and random factors
                base_density = max(0, 1.0 - abs(self.terrain_elevation[i, j] - 50) / 100)
                vegetation[i, j] = base_density * random.uniform(0.3, 1.0)
        return vegetation
    
    def _generate_water_bodies(self):
        """Generate lakes and rivers"""
        water_bodies = []
        
        # Add a lake
        lake_center = [self.width * 0.7, self.height * 0.3]
        lake_radius = 40
        water_bodies.append({
            'type': 'lake',
            'center': lake_center,
            'radius': lake_radius,
            'depth': 20
        })
        
        # Add a river (simplified as connected segments)
        river_points = [
            [50, self.height * 0.8],
            [150, self.height * 0.7],
            [250, self.height * 0.6],
            [350, self.height * 0.5]
        ]
        water_bodies.append({
            'type': 'river',
            'points': river_points,
            'width': 15,
            'depth': 5
        })
        
        return water_bodies
    
    def _generate_rich_obstacles(self):
        """Generate diverse obstacles with rich properties"""
        obstacles = []
        np.random.seed(42)
        
        obstacle_types = list(ObstacleType)
        
        for i in range(12):
            obs_type = random.choice(obstacle_types)
            x = np.random.randint(50, self.width-50)
            y = np.random.randint(50, self.height-50)
            
            # Initialize obstacle dictionary
            obstacle = {
                'pos': np.array([x, y]),
                'radius': 20,  # Default radius
                'height': 20,  # Default height
                'material': 'unknown',
                'color': [128, 128, 128],  # Default gray
                'type': obs_type
            }
            
            if obs_type == ObstacleType.ROCK:
                obstacle.update({
                    'radius': np.random.randint(20, 35),
                    'height': np.random.randint(15, 30),
                    'material': 'granite',
                    'color': [120, 120, 130],
                    'hardness': 0.9
                })
            elif obs_type == ObstacleType.TREE:
                obstacle.update({
                    'radius': np.random.randint(12, 20),
                    'height': np.random.randint(30, 60),
                    'material': 'wood',
                    'color': [34, 139, 34],
                    'leaves': random.choice(['oak', 'pine', 'willow']),
                    'age': np.random.randint(10, 100)
                })
            elif obs_type == ObstacleType.BUILDING:
                obstacle.update({
                    'radius': np.random.randint(25, 40),
                    'height': np.random.randint(40, 80),
                    'material': 'stone',
                    'color': [169, 169, 169],
                    'architecture': random.choice(['medieval', 'modern', 'fantasy']),
                    'condition': random.choice(['pristine', 'weathered', 'ruined'])
                })
            elif obs_type == ObstacleType.CRYSTAL:
                obstacle.update({
                    'radius': np.random.randint(8, 18),
                    'height': np.random.randint(20, 40),
                    'material': 'crystal',
                    'color': [random.randint(100, 255) for _ in range(3)],
                    'energy_level': random.uniform(0.3, 1.0),
                    'resonance_frequency': random.uniform(100, 1000)
                })
            elif obs_type == ObstacleType.WATER:
                obstacle.update({
                    'radius': np.random.randint(15, 25),
                    'height': 0,  # Water pools are at ground level
                    'material': 'water',
                    'color': [70, 130, 180],
                    'depth': random.uniform(1.0, 5.0),
                    'clarity': random.uniform(0.5, 1.0)
                })
            elif obs_type == ObstacleType.RUINS:
                obstacle.update({
                    'radius': np.random.randint(20, 35),
                    'height': np.random.randint(10, 25),
                    'material': 'weathered_stone',
                    'color': [139, 139, 131],
                    'age': random.randint(100, 1000),
                    'condition': 'ruined'
                })
            
            obstacles.append(obstacle)
        
        return obstacles

    
    def _generate_interactive_objects(self):
        """Generate objects that can be interacted with"""
        objects = []
        
        # Add some treasures, switches, portals, etc.
        for _ in range(5):
            obj_type = random.choice(['treasure', 'switch', 'portal', 'shrine'])
            x = np.random.randint(80, self.width-80)
            y = np.random.randint(80, self.height-80)
            
            obj = {
                'type': obj_type,
                'pos': np.array([x, y]),
                'radius': 10,
                'active': random.choice([True, False]),
                'interaction_count': 0
            }
            objects.append(obj)
        
        return objects
    
    def _update_time_and_weather(self):
        """Update world time and weather conditions"""
        # Advance time
        self.world_time += 0.1  # 6 minutes per step
        if self.world_time >= 24:
            self.world_time = 0
            self.day_count += 1
        
        # Occasionally change weather
        if random.random() < 0.01:  # 1% chance per step
            self.weather = random.choice(list(WeatherType))
            self.weather_intensity = random.uniform(0.2, 0.9)
        
        # Update temperature based on time and weather
        base_temp = 15 + 10 * math.sin((self.world_time - 6) * math.pi / 12)
        weather_modifier = {
            WeatherType.CLEAR: 0,
            WeatherType.RAINY: -5,
            WeatherType.FOGGY: -2,
            WeatherType.STORMY: -8,
            WeatherType.SNOWY: -15
        }
        self.temperature = base_temp + weather_modifier[self.weather]
        
        # Update ambient light
        if 6 <= self.world_time <= 18:  # Day
            self.ambient_light = 0.8 + 0.2 * math.sin((self.world_time - 6) * math.pi / 12)
        else:  # Night
            self.ambient_light = 0.1 + 0.1 * random.random()
        
        # Update wind
        self.wind_direction += random.uniform(-0.1, 0.1)
        self.wind_speed = max(0, self.wind_speed + random.uniform(-1, 1))
    
    def add_person(self, name: str, position: Tuple[float, float] = None, personality: str = None) -> str:
        """Add a person to the world with unique ID"""
        import uuid
        person_id = str(uuid.uuid4())
        
        if position is None:
            # Random position near agent
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(5, 20)
            pos_x = self.agent_pos[0] + distance * math.cos(angle)
            pos_y = self.agent_pos[1] + distance * math.sin(angle)
            position = (pos_x, pos_y)
        
        person = {
            'id': person_id,
            'name': name,
            'position': np.array(position),
            'personality': personality,
            'action': 'standing',
            'speech_content': '',
            'is_leaving': False,
            'target_position': None,
            'conversation_turns': 0
        }
        
        self.persons.append(person)
        return person_id
    
    def remove_person(self, person_id: str):
        """Remove a person from the world"""
        self.persons = [p for p in self.persons if p['id'] != person_id]
    
    def update_person_action(self, person_id: str, action: str, speech_content: str = ""):
        """Update a person's action and speech"""
        for person in self.persons:
            if person['id'] == person_id:
                person['action'] = action
                if speech_content:
                    person['speech_content'] = speech_content
                break
    
    def move_person_toward(self, person_id: str, target_position: Tuple[float, float], speed: float = 2.0):
        """Move a person toward a target position"""
        for person in self.persons:
            if person['id'] == person_id:
                direction = np.array(target_position) - person['position']
                distance = np.linalg.norm(direction)
                
                if distance > 1.0:  # Not at target yet
                    direction = direction / distance
                    person['position'] += direction * min(speed, distance)
                    person['action'] = 'walking'
                else:
                    person['position'] = np.array(target_position)
                    person['action'] = 'standing'
                    person['target_position'] = None
                break
    
    def get_person_distance(self, person_id: str) -> float:
        """Get distance from agent to person"""
        for person in self.persons:
            if person['id'] == person_id:
                return np.linalg.norm(person['position'] - self.agent_pos)
        return float('inf')
    
    def get_nearby_persons(self, max_distance: float = 50.0) -> List[Dict]:
        """Get all persons within max_distance"""
        nearby = []
        for person in self.persons:
            distance = np.linalg.norm(person['position'] - self.agent_pos)
            if distance <= max_distance:
                person_info = person.copy()
                person_info['distance'] = distance
                nearby.append(person_info)
        return nearby
    
    def _get_time_of_day(self):
        """Determine time of day category"""
        if 5 <= self.world_time < 7:
            return TimeOfDay.DAWN
        elif 7 <= self.world_time < 11:
            return TimeOfDay.MORNING
        elif 11 <= self.world_time < 14:
            return TimeOfDay.NOON
        elif 14 <= self.world_time < 18:
            return TimeOfDay.AFTERNOON
        elif 18 <= self.world_time < 20:
            return TimeOfDay.DUSK
        else:
            return TimeOfDay.NIGHT
    
    def step(self, action):
        """Step simulation with enhanced environmental updates"""
        forward_speed, turn_rate = action
        self.agent_angle += turn_rate * 0.1
        
        # Update position with environmental effects
        base_speed = forward_speed * 3
        
        # Wind effect
        wind_effect_x = self.wind_speed * 0.1 * math.cos(self.wind_direction)
        wind_effect_y = self.wind_speed * 0.1 * math.sin(self.wind_direction)
        
        dx = base_speed * math.cos(self.agent_angle) + wind_effect_x
        dy = base_speed * math.sin(self.agent_angle) + wind_effect_y
        
        new_pos = self.agent_pos + np.array([dx, dy])
        
        # Enhanced boundary checking with terrain
        new_pos[0] = np.clip(new_pos[0], 20, self.width-20)
        new_pos[1] = np.clip(new_pos[1], 20, self.height-20)
        
        # Check for water collision
        for water_body in self.water_bodies:
            if water_body['type'] == 'lake':
                dist = np.linalg.norm(new_pos - water_body['center'])
                if dist < water_body['radius']:
                    self.agent_health -= 5  # Swimming is tiring
        
        self.agent_pos = new_pos
        self.step_count += 1
        
        # Update world state
        self._update_time_and_weather()
        
        # Update agent status
        self.agent_energy = max(0, self.agent_energy - 0.5)
        if self.agent_energy == 0:
            self.agent_health -= 1
        
        return self._get_enhanced_sensory_data()
    
    def _get_enhanced_sensory_data(self):
        """Extract comprehensive sensory data with rich environmental information"""
        visual_array = self._render_to_array()
        proximity_sensors = self._get_proximity_readings()
        
        # Get current elevation
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        current_elevation = self.terrain_elevation[agent_y, agent_x] if 0 <= agent_x < self.width and 0 <= agent_y < self.height else 0
        
        sensory_data = {
            'timestamp': self.step_count,
            'world_time': self.world_time,
            'day_count': self.day_count,
            
            'visual': {
                'rgb_camera': visual_array,
                'resolution': (self.width, self.height),
                'ambient_light': self.ambient_light,
                'visibility_range': self._calculate_visibility(),
                'dominant_colors': self._analyze_scene_colors()
            },
            
            'spatial': {
                'position': self.agent_pos.copy(),
                'elevation': current_elevation,
                'heading': self.agent_angle,
                'velocity': np.array([math.cos(self.agent_angle), math.sin(self.agent_angle)]),
                'speed': 1.0,
                'terrain_slope': self._get_terrain_slope()
            },
            
            'proximity': proximity_sensors,
            'proximity_detailed': self._get_detailed_proximity(),
            
            'proprioceptive': {
                'health': self.agent_health,
                'energy': self.agent_energy,
                'steering_angle': 0.0,
                'acceleration': 0.0,
                'balance': 1.0 - abs(self._get_terrain_slope()) * 0.1
            },
            
            'environmental': {
                'time_of_day': self._get_time_of_day().value,
                'weather': {
                    'condition': self.weather.value,
                    'intensity': self.weather_intensity,
                    'temperature': self.temperature,
                    'wind_speed': self.wind_speed,
                    'wind_direction': self.wind_direction
                },
                'atmosphere': {
                    'humidity': self._calculate_humidity(),
                    'air_pressure': 1013.25 + random.uniform(-20, 20),
                    'air_quality': self._calculate_air_quality()
                },
                'terrain': {
                    'type': self._identify_terrain_type(),
                    'elevation': current_elevation,
                    'vegetation_density': self._get_local_vegetation(),
                    'ground_composition': self._analyze_ground()
                },
                'nearby_features': self._detect_nearby_features(),
                'acoustic_environment': self._generate_sound_landscape()
            }
        }
        
        return sensory_data
    
    def _calculate_visibility(self):
        """Calculate visibility range based on weather and time"""
        base_visibility = 100
        
        weather_modifiers = {
            WeatherType.CLEAR: 1.0,
            WeatherType.RAINY: 0.7,
            WeatherType.FOGGY: 0.3,
            WeatherType.STORMY: 0.5,
            WeatherType.SNOWY: 0.4
        }
        
        time_modifier = self.ambient_light
        weather_modifier = weather_modifiers[self.weather]
        
        return base_visibility * time_modifier * weather_modifier
    
    def _analyze_scene_colors(self):
        """Analyze dominant colors in the scene"""
        colors = {
            'sky': self._get_sky_color(),
            'ground': self._get_ground_color(),
            'vegetation': [34, 139, 34],
            'water': [70, 130, 180]
        }
        return colors
    
    def _get_sky_color(self):
        """Get sky color based on time and weather"""
        time_of_day = self._get_time_of_day()
        
        if time_of_day == TimeOfDay.DAWN:
            return [255, 165, 0]  # Orange
        elif time_of_day == TimeOfDay.MORNING:
            return [135, 206, 235]  # Sky blue
        elif time_of_day == TimeOfDay.NOON:
            return [87, 151, 255]  # Bright blue
        elif time_of_day == TimeOfDay.AFTERNOON:
            return [135, 206, 235]  # Sky blue
        elif time_of_day == TimeOfDay.DUSK:
            return [255, 69, 0]  # Red-orange
        else:  # Night
            return [25, 25, 112]  # Midnight blue
    
    def _get_ground_color(self):
        """Get ground color based on terrain and moisture"""
        base_color = [139, 69, 19]  # Brown earth
        
        if self.weather in [WeatherType.RAINY, WeatherType.STORMY]:
            # Darker, more saturated when wet
            return [int(c * 0.7) for c in base_color]
        elif self.weather == WeatherType.SNOWY:
            return [240, 248, 255]  # Snow white
        
        return base_color
    
    def _get_detailed_proximity(self):
        """Get detailed proximity information including persons"""
        detailed = []
        
        # Add person information
        for person in self.persons:
            distance = np.linalg.norm(person['position'] - self.agent_pos)
            if distance <= 50:  # Only include nearby persons
                direction = person['position'] - self.agent_pos
                angle = math.atan2(direction[1], direction[0])
                
                person_info = {
                    'type': 'person',
                    'person_id': person['id'],
                    'name': person['name'],
                    'distance': distance,
                    'direction': angle,
                    'action': person['action'],
                    'speech_content': person['speech_content'],
                    'position': person['position'].tolist()
                }
                detailed.append(person_info)
        
        # Add obstacle information
        for obs in self.obstacles:
            distance = np.linalg.norm(obs['pos'] - self.agent_pos)
            if distance <= 30:
                direction = obs['pos'] - self.agent_pos
                angle = math.atan2(direction[1], direction[0])
                
                obs_info = {
                    'type': 'obstacle',
                    'obstacle_type': obs['type'].value,
                    'distance': distance,
                    'direction': angle,
                    'material': obs.get('material', 'unknown'),
                    'position': obs['pos'].tolist()
                }
                detailed.append(obs_info)
        
        return detailed
    
    def _render_to_array(self):
        """Create visual array representation for compatibility"""
        # Simple placeholder - return a basic visual array
        import numpy as np
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def _get_proximity_readings(self):
        """Get proximity sensor readings"""
        # Simple placeholder for proximity sensors
        return {}
    
    def _get_terrain_slope(self):
        """Calculate terrain slope at current position"""
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if x >= self.width - 1 or y >= self.height - 1 or x < 1 or y < 1:
            return 0.0
        # Simple slope calculation
        return 0.0  # Placeholder
    
    def _calculate_humidity(self):
        """Calculate humidity"""
        return 50.0  # Placeholder
    
    def _calculate_air_quality(self):
        """Calculate air quality"""
        return 80.0  # Placeholder
    
    def _get_local_vegetation(self):
        """Get local vegetation density"""
        return 0.5  # Placeholder
    
    def _analyze_ground(self):
        """Analyze ground composition"""
        return "soil"  # Placeholder
    
    def _identify_terrain_type(self):
        """Identify terrain type"""
        return "urban"  # Placeholder
    
    def _detect_nearby_features(self):
        """Detect nearby features"""
        return []  # Placeholder
    
    def _generate_sound_landscape(self):
        """Generate sound landscape"""
        return []  # Placeholder
    
    def render(self):
        """Enhanced visualization with environmental information"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        
        # Draw terrain elevation as background
        extent = [0, self.width, 0, self.height]
        self.ax.imshow(self.terrain_elevation, extent=extent, alpha=0.3, cmap='terrain', origin='lower')
        
        # Draw water bodies
        for water_body in self.water_bodies:
            if water_body['type'] == 'lake':
                circle = Circle(water_body['center'], water_body['radius'], 
                              color='blue', alpha=0.6, label='Lake')
                self.ax.add_patch(circle)
        
        # Draw obstacles with different colors based on type
        for obs in self.obstacles:
            color_map = {
                ObstacleType.ROCK: 'gray',
                ObstacleType.TREE: 'green',
                ObstacleType.BUILDING: 'brown',
                ObstacleType.CRYSTAL: 'purple',
                ObstacleType.WATER: 'blue',
                ObstacleType.RUINS: 'darkgray'
            }
            color = color_map.get(obs['type'], 'black')
            circle = Circle(obs['pos'], obs['radius'], color=color, alpha=0.7)
            self.ax.add_patch(circle)
        
        # Draw interactive objects
        for obj in self.interactive_objects:
            color = 'gold' if obj['active'] else 'silver'
            circle = Circle(obj['pos'], obj['radius'], color=color, alpha=0.8)
            self.ax.add_patch(circle)
        
        # Draw agent
        agent_circle = Circle(self.agent_pos, 8, color='red', alpha=0.9)
        self.ax.add_patch(agent_circle)
        
        # Draw heading direction
        end_pos = self.agent_pos + 25 * np.array([math.cos(self.agent_angle), math.sin(self.agent_angle)])
        self.ax.arrow(self.agent_pos[0], self.agent_pos[1], 
                     end_pos[0] - self.agent_pos[0], end_pos[1] - self.agent_pos[1],
                     head_width=8, head_length=8, fc='red', ec='red')
        
        # Enhanced title with environmental info
        time_str = f"{int(self.world_time):02d}:{int((self.world_time % 1) * 60):02d}"
        title = (f"Step: {self.step_count} | Time: {time_str} | Weather: {self.weather.value.capitalize()} | "
                f"Temp: {self.temperature:.1f}°C | Health: {self.agent_health:.1f} | Energy: {self.agent_energy:.1f}")
        plt.title(title, fontsize=10)
        
        plt.pause(0.05)
        plt.show()

# Usage example:
if __name__ == "__main__":
    world = SimpleWorld(600, 400)
    
    # Run a few simulation steps
    for step in range(10):
        action = [random.uniform(-1, 1), random.uniform(-0.5, 0.5)]
        sensory_data = world.step(action)
        world.render()
        
        # Print some interesting sensory data
        print(f"\nStep {step}:")
        print(f"Time: {sensory_data['world_time']:.1f} ({sensory_data['environmental']['time_of_day']})")
        print(f"Weather: {sensory_data['environmental']['weather']['condition']} "
              f"({sensory_data['environmental']['weather']['intensity']:.2f})")
        print(f"Terrain: {sensory_data['environmental']['terrain']['type']} "
              f"(elevation: {sensory_data['environmental']['terrain']['elevation']:.1f})")
        print(f"Nearby features: {len(sensory_data['environmental']['nearby_features'])}")
