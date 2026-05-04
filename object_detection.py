

import numpy as np
import cv2
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# ARC-AGI-3 Color Palette (0-15)
# ============================================

class ARCColor:
    """Color indices for ARC-AGI-3 16-color palette"""
    BLACK = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    BLUE = 4
    YELLOW = 5
    CYAN = 6
    MAGENTA = 7
    BROWN = 8
    PINK = 9
    DARK_GREEN = 10
    ORANGE = 11
    PURPLE = 12
    LIGHT_BLUE = 13
    LIGHT_GREEN = 14
    LIGHT_GRAY = 15

# Map color indices to human-readable names
COLOR_NAMES = {
    ARCColor.BLACK: "background",
    ARCColor.WHITE: "cursor/agent",
    ARCColor.RED: "end_goal",
    ARCColor.GREEN: "start_point",
    ARCColor.BLUE: "water/obstacle",
    ARCColor.YELLOW: "waypoint",
    ARCColor.CYAN: "path",
    ARCColor.MAGENTA: "enemy",
    ARCColor.BROWN: "wall",
    ARCColor.PINK: "reward",
    ARCColor.DARK_GREEN: "forest",
    ARCColor.ORANGE: "key",
    ARCColor.PURPLE: "door",
    ARCColor.LIGHT_BLUE: "path",
    ARCColor.LIGHT_GREEN: "grass",
    ARCColor.LIGHT_GRAY: "neutral"
}

# ============================================
# TRADITIONAL COMPUTER VISION DETECTOR
# ============================================

class TraditionalObjectDetector:
    """
    Detects objects in ARC-AGI-3 grids using color-based segmentation.
    This is the fastest and most reliable method for ARC games.
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.min_object_size = 2  # Minimum pixels to consider as object

    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect objects in the frame.

        Args:
            frame: 64x64 numpy array of color indices (0-15)

        Returns:
            Dictionary mapping object type to list of (x, y) positions
        """
        objects = {
            'agent': [],
            'start': [],
            'end': [],
            'waypoints': [],
            'path': [],
            'walls': [],
            'keys': [],
            'doors': [],
            'rewards': []
        }

        # Method 1: Direct color lookup (fastest)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = frame[y, x]

                if color == ARCColor.WHITE:
                    objects['agent'].append((x, y))
                elif color == ARCColor.GREEN:
                    objects['start'].append((x, y))
                elif color == ARCColor.RED:
                    objects['end'].append((x, y))
                elif color == ARCColor.YELLOW:
                    objects['waypoints'].append((x, y))
                elif color in [ARCColor.CYAN, ARCColor.LIGHT_BLUE]:
                    objects['path'].append((x, y))
                elif color == ARCColor.BLACK:
                    continue
                elif color == ARCColor.BROWN:
                    objects['walls'].append((x, y))
                elif color == ARCColor.ORANGE:
                    objects['keys'].append((x, y))
                elif color == ARCColor.PURPLE:
                    objects['doors'].append((x, y))
                elif color == ARCColor.PINK:
                    objects['rewards'].append((x, y))

        return objects

    def find_connected_components(self, frame: np.ndarray, target_color: int) -> List[List[Tuple[int, int]]]:
        """
        Find connected components of a specific color using flood fill.
        Useful for grouping adjacent pixels into distinct objects.
        """
        visited = np.zeros_like(frame, dtype=bool)
        components = []

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if frame[y, x] == target_color and not visited[y, x]:
                    # Flood fill to find connected component
                    component = []
                    stack = [(x, y)]

                    while stack:
                        cx, cy = stack.pop()
                        if visited[cy, cx]:
                            continue
                        visited[cy, cx] = True
                        component.append((cx, cy))

                        # Check 4-connected neighbors
                        for dx, dy in [(1 ,0), (-1 ,0), (0 ,1), (0 ,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                if frame[ny, nx] == target_color and not visited[ny, nx]:
                                    stack.append((nx, ny))

                    if len(component) >= self.min_object_size:
                        components.append(component)

        return components

    def get_object_centroids(self, objects: Dict) -> Dict[str, List[Tuple[float, float]]]:
        """Calculate centroids of detected objects"""
        centroids = {}

        for obj_type, positions in objects.items():
            if positions:
                # Group positions into components if needed
                # For now, just calculate average
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                centroids[obj_type] = [(sum(xs ) /len(xs), sum(ys ) /len(ys))]
            else:
                centroids[obj_type] = []

        return centroids


# ============================================
# DEEP LEARNING DETECTOR (CNN)
# ============================================

class SimpleObjectCNN(nn.Module):
    """
    Simple CNN for detecting objects in ARC-AGI-3 frames.
    This approach learns to recognize patterns without explicit color rules.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input: (batch, 1, 64, 64)
        x = self.pool(F.relu(self.conv1(x)))   # 64->32
        x = self.pool(F.relu(self.conv2(x)))   # 32->16
        x = self.pool(F.relu(self.conv3(x)))   # 16->8

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CNNObjectDetector:
    """
    Wrapper for CNN-based object detection.
    """

    def __init__(self, model_path: Optional[str] = None, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleObjectCNN(num_classes).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect objects using sliding window approach.
        """
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)

        # Sliding window detection
        window_size = 16
        stride = 8
        detections = defaultdict(list)

        for y in range(0, 64 - window_size + 1, stride):
            for x in range(0, 64 - window_size + 1, stride):
                window = frame_tensor[:, :, y: y +window_size, x: x +window_size]

                with torch.no_grad():
                    logits = self.model(window)
                    probs = F.softmax(logits, dim=1)

                    max_prob, pred_class = torch.max(probs, 1)

                    if max_prob > 0.7:  # Confidence threshold
                        center_x = x + window_size // 2
                        center_y = y + window_size // 2
                        detections[pred_class.item()].append((center_x, center_y))

        # Convert to human-readable names (mapping depends on training)
        object_map = {0: 'agent', 1: 'waypoint', 2: 'goal', 3: 'obstacle', 4: 'path'}

        result = {}
        for class_id, positions in detections.items():
            key = object_map.get(class_id, f'class_{class_id}')
            result[key] = positions

        return result


# ============================================
# HYBRID DETECTOR (Best of Both Worlds)
# ============================================

class HybridObjectDetector:
    """
    Combines color-based detection with the CNN for maximum accuracy.
    """

    def __init__(self, cnn_model_path: Optional[str] = None):
        self.color_detector = TraditionalObjectDetector()
        self.cnn_detector = CNNObjectDetector(cnn_model_path) if cnn_model_path else None

    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect objects using both methods and merge results.
        """
        # Get color-based detections (fast and accurate for standard colors)
        color_detections = self.color_detector.detect_objects(frame)

        # If CNN is available, use it for ambiguous cases
        if self.cnn_detector:
            cnn_detections = self.cnn_detector.detect_objects(frame)

            # Merge: prefer color detections for standard objects,
            # but use CNN for things that might have variable colors
            merged = color_detections.copy()
            for obj_type, positions in cnn_detections.items():
                if obj_type not in merged or not merged[obj_type]:
                    merged[obj_type] = positions

            return merged

        return color_detections


# ============================================
# INTEGRATION WITH ARC-AGI-3 ENVIRONMENT
# ============================================

class ARCObservationParser:
    """
    Parse ARC-AGI-3 environment observations into structured game state.
    """

    def __init__(self, use_cnn: bool = False, cnn_model_path: Optional[str] = None):
        self.detector = HybridObjectDetector(cnn_model_path) if use_cnn else TraditionalObjectDetector()

    def parse_observation(self, obs) -> Dict:
        """
        Parse frame from ARC-AGI-3 environment.

        Args:
            obs: FrameDataRaw object from env.reset() or env.step()

        Returns:
            Structured game state
        """
        # Extract grid
        if hasattr(obs, 'frame'):
            frame = obs.frame
        elif hasattr(obs, 'grid'):
            frame = obs.grid
        else:
            raise ValueError("Observation has no 'frame' or 'grid' attribute")

        # Detect objects
        objects = self.detector.detect_objects(frame)

        # Extract metadata
        metadata = {
            'state': getattr(obs, 'state', None),
            'levels_completed': getattr(obs, 'levels_completed', 0),
            'win_levels': getattr(obs, 'win_levels', 0)
        }

        # Build structured state
        game_state = {
            'frame': frame,
            'objects': objects,
            'agent_pos': objects.get('agent', [None])[0],
            'start_pos': objects.get('start', [None])[0],
            'end_pos': objects.get('end', [None])[0],
            'waypoints': objects.get('waypoints', []),
            'path': objects.get('path', []),
            'walls': objects.get('walls', []),
            'keys': objects.get('keys', []),
            'doors': objects.get('doors', []),
            'rewards': objects.get('rewards', []),
            'metadata': metadata
        }

        return game_state


# ============================================
# EXAMPLE USAGE
# ============================================

def demo_detection():
    """Demonstrate object detection on a synthetic frame"""

    # Create synthetic 64x64 frame
    frame = np.full((64, 64), ARCColor.BLACK, dtype=np.uint8)

    # Add some objects
    frame[10, 10] = ARCColor.GREEN      # Start
    frame[50, 50] = ARCColor.RED        # End
    frame[20, 20] = ARCColor.YELLOW     # Waypoint
    frame[30, 30] = ARCColor.YELLOW     # Another waypoint
    frame[15, 10] = ARCColor.WHITE      # Agent
    frame[25:30, 40:45] = ARCColor.CYAN # Path

    # Initialize detector
    detector = TraditionalObjectDetector()

    # Detect objects
    objects = detector.detect_objects(frame)

    print("=== Detected Objects ===")
    for obj_type, positions in objects.items():
        if positions:
            print(f"{obj_type}: {positions[:5]}..." if len(positions) > 5 else f"{obj_type}: {positions}")

    # Calculate centroids
    centroids = detector.get_object_centroids(objects)
    print("\n=== Object Centroids ===")
    for obj_type, centroid in centroids.items():
        if centroid:
            print(f"{obj_type}: {centroid[0]}")


def integrate_with_env():
    """
    Example of integrating with arc-witness-envs environment.
    """
    # This assumes you have arc-witness-envs installed
    # from arc_agi import Arcade
    # from openenv_adapter.client import WitnessEnvClient

    # Initialize parser
    parser = ARCObservationParser(use_cnn=False)

    # Example loop (pseudocode)
    print("\n=== Integration Example ===")
    print("""
    # In your agent loop:

    from arc_agi import Arcade

    arc = Arcade()
    env = arc.make("tw01")  # PathDots game

    obs = env.reset()
    while not done:
        game_state = parser.parse_observation(obs)

        # Use structured state for decision making
        agent_pos = game_state['agent_pos']
        waypoints = game_state['waypoints']

        if agent_pos and waypoints:
            target = waypoints[0]  # Go to first waypoint
            # ... decide action based on target ...

        # ... execute action ...
        obs = env.step(action)
        """)


