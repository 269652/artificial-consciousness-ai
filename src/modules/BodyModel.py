from typing import Optional, Tuple

class BodyModel:
    """Virtual body model exposing executable actions for the PFC."""

    def __init__(self):
        self.position: Tuple[float, float] = (0.0, 0.0)  # Current (x, y) position
        self.heading: float = 0.0  # Heading in radians

    def navigate(self, direction: str, distance: float) -> bool:
        """Navigate in a direction by a distance. Args: direction (str: 'north', 'south', etc.), distance (float: meters). Returns: success (bool)."""
        print(f"Navigating {direction} by {distance} meters.")
        # Simulate movement (update position)
        if direction == "north":
            self.position = (self.position[0], self.position[1] + distance)
        elif direction == "south":
            self.position = (self.position[0], self.position[1] - distance)
        elif direction == "east":
            self.position = (self.position[0] + distance, self.position[1])
        elif direction == "west":
            self.position = (self.position[0] - distance, self.position[1])
        return True  # Assume success

    def push_object(self, object_id: str, force: float, direction: str) -> bool:
        """Push an object with force in a direction. Args: object_id (str), force (float: Newtons), direction (str: 'forward', etc.). Returns: success (bool)."""
        print(f"Pushing object {object_id} with {force} N in {direction}.")
        return True  # Simulate success

    def hit_object(self, object_id: str, force: float) -> bool:
        """Hit an object with force. Args: object_id (str), force (float: Newtons). Returns: success (bool)."""
        print(f"Hitting object {object_id} with {force} N.")
        return True  # Simulate success

    def talk(self, message: str, target: Optional[str] = None) -> bool:
        """Speak a message to a target. Args: message (str), target (Optional[str]: person ID or None for general). Returns: success (bool)."""
        if target:
            print(f"Talking to {target}: {message}")
        else:
            print(f"Speaking: {message}")
        return True  # Simulate success

    def jump(self, height: float) -> bool:
        """Jump to a height. Args: height (float: meters). Returns: success (bool)."""
        print(f"Jumping to {height} meters.")
        return True  # Simulate success

    def wave(self, direction: str) -> bool:
        """Wave in a direction. Args: direction (str: 'left', 'right', etc.). Returns: success (bool)."""
        print(f"Waving {direction}.")
        return True  # Simulate success

    def yell(self, message: str) -> bool:
        """Yell a message. Args: message (str). Returns: success (bool)."""
        print(f"Yelling: {message.upper()}")
        return True  # Simulate success
