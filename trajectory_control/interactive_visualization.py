"""
NATIVE INTERACTIVE VISUALIZATION SYSTEM
Enhanced OpenGL-based 2D, 3D, and 4D Interactive Viewers

Features:
- Multi-window support
- Real-time data streaming
- Enhanced UI with themes
- Export capabilities
- Performance optimizations
- Better camera controls
- Conflict simulation and analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import json
import pickle
import threading
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

# OpenGL and PyGame for native graphics
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *  # Added for 3D shapes
    OPENGL_AVAILABLE = True
except ImportError:
    print("❌ PyGame/OpenGL not installed!")
    print("Install with: pip install pygame PyOpenGL PyOpenGL_accelerate")
    OPENGL_AVAILABLE = False
    exit(1)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Waypoint:
    """Enhanced waypoint with velocity and acceleration"""
    x: float
    y: float
    z: float
    t: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    id: int = 0
    
    @classmethod
    def from_any(cls, data):
        """Create Waypoint from any data format"""
        if isinstance(data, cls):
            # Already a Waypoint object
            return data
        elif isinstance(data, dict):
            # Dictionary format
            return cls(
                x=data.get('x', 0),
                y=data.get('y', 0),
                z=data.get('z', 0),
                t=data.get('t', 0),
                vx=data.get('vx', 0),
                vy=data.get('vy', 0),
                vz=data.get('vz', 0)
            )
        elif hasattr(data, 'x') and hasattr(data, 'y') and hasattr(data, 'z') and hasattr(data, 't'):
            # Object with x, y, z, t attributes (like Waypoint4D)
            return cls(
                x=data.x,
                y=data.y,
                z=data.z,
                t=data.t,
                vx=getattr(data, 'vx', 0),
                vy=getattr(data, 'vy', 0),
                vz=getattr(data, 'vz', 0)
            )
        elif isinstance(data, (list, tuple)):
            # Tuple/list format (x, y, z, t, ...)
            return cls(
                x=data[0] if len(data) > 0 else 0,
                y=data[1] if len(data) > 1 else 0,
                z=data[2] if len(data) > 2 else 0,
                t=data[3] if len(data) > 3 else 0,
                vx=data[4] if len(data) > 4 else 0,
                vy=data[5] if len(data) > 5 else 0,
                vz=data[6] if len(data) > 6 else 0
            )
        else:
            # Unknown format, return default
            print(f"⚠️  Warning: Unknown waypoint format: {type(data)}")
            return cls(0, 0, 0, 0)
    
@dataclass 
class Drone:
    """Drone with complete state information"""
    id: str
    waypoints: List[Waypoint]
    color: Tuple[float, float, float] = (0.3, 0.5, 1.0)
    radius: float = 1.0
    is_primary: bool = False
    show_trajectory: bool = True
    show_labels: bool = True
    
    @classmethod
    def from_any(cls, drone_id: str, data: Any):
        """Create Drone from any data format"""
        waypoints = []
        
        if hasattr(data, 'waypoints'):
            # Object with waypoints attribute
            for wp_data in data.waypoints:
                waypoints.append(Waypoint.from_any(wp_data))
        elif isinstance(data, dict) and 'waypoints' in data:
            # Dictionary with waypoints key
            for wp_data in data['waypoints']:
                waypoints.append(Waypoint.from_any(wp_data))
        elif isinstance(data, (list, tuple)):
            # Direct list of waypoints
            for wp_data in data:
                waypoints.append(Waypoint.from_any(wp_data))
        else:
            # Unknown format
            print(f"⚠️  Warning: Unknown drone data format for {drone_id}: {type(data)}")
        
        # Get other properties
        if isinstance(data, dict):
            color = data.get('color', (0.3, 0.5, 1.0))
            is_primary = data.get('is_primary', False)
            radius = data.get('radius', 1.0)
        else:
            # Try to get attributes from object
            color = getattr(data, 'color', (0.3, 0.5, 1.0))
            is_primary = getattr(data, 'is_primary', False)
            radius = getattr(data, 'radius', 1.0)
        
        return cls(
            id=drone_id,
            waypoints=waypoints,
            color=color,
            is_primary=is_primary,
            radius=radius
        )
        
@dataclass
class Conflict:
    """Conflict with enhanced information"""
    time: float
    location: Dict[str, float]
    severity: str
    drones_involved: List[str]
    distance: float
    type: str = "proximity"
    resolved: bool = False
    risk_score: float = 0.0
    
    def __post_init__(self):
        # Calculate risk score based on severity
        severity_scores = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25
        }
        self.risk_score = severity_scores.get(self.severity, 0.0)
        
        # Adjust based on number of drones
        self.risk_score *= min(1.0 + 0.2 * len(self.drones_involved), 2.0)
    
    @classmethod
    def from_any(cls, data: Any):
        """Create Conflict from any data format"""
        if isinstance(data, cls):
            # Already a Conflict object
            return data
            
        # Initialize with defaults
        time_val = 0
        location_val = {'x': 0, 'y': 0, 'z': 0}
        severity_val = 'MEDIUM'
        drones_involved_val = []
        distance_val = 10.0
        type_val = 'proximity'
        
        if isinstance(data, dict):
            # Dictionary format
            time_val = data.get('time', 0)
            severity_val = data.get('severity', 'MEDIUM')
            drones_involved_val = data.get('drones_involved', [])
            distance_val = data.get('distance', 10.0)
            type_val = data.get('type', 'proximity')
            
            # Handle location in various formats
            loc_data = data.get('location', {})
            if isinstance(loc_data, dict):
                location_val = loc_data
            elif hasattr(loc_data, 'x') and hasattr(loc_data, 'y') and hasattr(loc_data, 'z'):
                location_val = {'x': loc_data.x, 'y': loc_data.y, 'z': loc_data.z}
            elif isinstance(loc_data, (list, tuple)) and len(loc_data) >= 3:
                location_val = {'x': loc_data[0], 'y': loc_data[1], 'z': loc_data[2]}
        else:
            # Try to get attributes from object
            time_val = getattr(data, 'time', 0)
            severity_val = getattr(data, 'severity', 'MEDIUM')
            drones_involved_val = getattr(data, 'drones_involved', [])
            distance_val = getattr(data, 'distance', 10.0)
            type_val = getattr(data, 'type', 'proximity')
            
            # Handle location
            loc_data = getattr(data, 'location', None)
            if loc_data:
                if isinstance(loc_data, dict):
                    location_val = loc_data
                elif hasattr(loc_data, 'x') and hasattr(loc_data, 'y') and hasattr(loc_data, 'z'):
                    location_val = {'x': loc_data.x, 'y': loc_data.y, 'z': loc_data.z}
                elif isinstance(loc_data, (list, tuple)) and len(loc_data) >= 3:
                    location_val = {'x': loc_data[0], 'y': loc_data[1], 'z': loc_data[2]}
        
        return cls(
            time=time_val,
            location=location_val,
            severity=severity_val,
            drones_involved=drones_involved_val,
            distance=distance_val,
            type=type_val
        )

class ViewMode(Enum):
    """Available view modes"""
    XY = "XY (Top View)"
    XZ = "XZ (Front View)"
    YZ = "YZ (Side View)"
    TIME_ALTITUDE = "Time-Altitude"
    TIME_DISTANCE = "Time-Distance"
    SPEED_PROFILE = "Speed Profile"

class Theme(Enum):
    """Visual themes"""
    LIGHT = "Light"
    DARK = "Dark"
    NIGHT_VISION = "Night Vision"
    BLUE_DARK = "Blue Dark"
    
    def get_colors(self):
        """Get color scheme for theme"""
        if self == self.LIGHT:
            return {
                'background': (0.95, 0.95, 0.95, 1.0),
                'grid': (0.8, 0.8, 0.8, 0.5),
                'text': (0.1, 0.1, 0.1, 1.0),
                'primary': (0.0, 0.4, 0.8, 1.0),
                'secondary': (0.8, 0.2, 0.2, 1.0),
                'highlight': (0.1, 0.7, 0.1, 1.0)
            }
        elif self == self.DARK:
            return {
                'background': (0.1, 0.1, 0.15, 1.0),
                'grid': (0.3, 0.3, 0.3, 0.3),
                'text': (0.9, 0.9, 0.9, 1.0),
                'primary': (0.2, 0.6, 1.0, 1.0),
                'secondary': (1.0, 0.4, 0.4, 1.0),
                'highlight': (0.4, 0.9, 0.4, 1.0)
            }
        elif self == self.NIGHT_VISION:
            return {
                'background': (0.0, 0.1, 0.0, 1.0),
                'grid': (0.0, 0.3, 0.0, 0.3),
                'text': (0.0, 1.0, 0.0, 1.0),
                'primary': (0.0, 1.0, 0.0, 1.0),
                'secondary': (1.0, 0.5, 0.0, 1.0),
                'highlight': (1.0, 1.0, 0.0, 1.0)
            }
        else:  # BLUE_DARK
            return {
                'background': (0.05, 0.05, 0.1, 1.0),
                'grid': (0.1, 0.1, 0.2, 0.3),
                'text': (0.8, 0.9, 1.0, 1.0),
                'primary': (0.0, 0.7, 1.0, 1.0),
                'secondary': (1.0, 0.3, 0.3, 1.0),
                'highlight': (0.3, 1.0, 0.3, 1.0)
            }

# ============================================================================
# DATA CONVERSION UTILITIES
# ============================================================================

def convert_to_drone_objects(drones_data: Dict) -> Dict[str, Drone]:
    """Convert any drone data format to Drone objects"""
    drones = {}
    
    for drone_id, data in drones_data.items():
        if isinstance(data, Drone):
            # Already a Drone object
            drones[drone_id] = data
        else:
            # Convert from any format
            drones[drone_id] = Drone.from_any(drone_id, data)
    
    return drones

def convert_to_conflict_objects(conflicts_data: List) -> List[Conflict]:
    """Convert any conflicts data format to Conflict objects"""
    conflicts = []
    
    for conflict_data in conflicts_data:
        if isinstance(conflict_data, Conflict):
            # Already a Conflict object
            conflicts.append(conflict_data)
        else:
            # Convert from any format
            conflicts.append(Conflict.from_any(conflict_data))
    
    return conflicts

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

class DisplayListCache:
    """Cache OpenGL display lists for performance"""
    def __init__(self):
        self.cache = {}
        self.enabled = True
        
    def get(self, key: str, generator_func):
        """Get or create display list"""
        if not self.enabled:
            return generator_func()
            
        if key not in self.cache:
            list_id = glGenLists(1)
            glNewList(list_id, GL_COMPILE)
            generator_func()
            glEndList()
            self.cache[key] = list_id
            
        glCallList(self.cache[key])
        
    def clear(self):
        """Clear cache"""
        for list_id in self.cache.values():
            glDeleteLists(list_id, 1)
        self.cache.clear()

class VertexBuffer:
    """Simple vertex buffer for batch rendering"""
    def __init__(self):
        self.vertices = []
        self.colors = []
        
    def add_vertex(self, x, y, z, r, g, b, a=1.0):
        """Add vertex with color"""
        self.vertices.extend([x, y, z])
        self.colors.extend([r, g, b, a])
        
    def draw(self, mode=GL_LINES):
        """Draw all vertices"""
        if not self.vertices:
            return
            
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        vertices_np = np.array(self.vertices, dtype=np.float32)
        colors_np = np.array(self.colors, dtype=np.float32)
        
        glVertexPointer(3, GL_FLOAT, 0, vertices_np)
        glColorPointer(4, GL_FLOAT, 0, colors_np)
        glDrawArrays(mode, 0, len(self.vertices) // 3)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
    def clear(self):
        """Clear buffer"""
        self.vertices.clear()
        self.colors.clear()

# ============================================================================
# BASE VIEWER CLASS
# ============================================================================

class BaseViewer:
    """Base class for all viewers with common functionality"""
    
    def __init__(self, title="Interactive Viewer", size=(1400, 900)):
        self.display = size
        self.title = title
        self.running = False
        
        # Performance
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 60
        
        # Camera
        self.camera_target = [0, 0, 0]
        self.camera_position = [0, 0, -100]
        self.camera_up = [0, 1, 0]
        
        # Theme
        self.theme = Theme.DARK
        self.colors = self.theme.get_colors()
        
        # Cache
        self.cache = DisplayListCache()
        self.vertex_buffer = VertexBuffer()
        
        # UI State
        self.show_ui = True
        self.show_stats = True
        self.fullscreen = False
        
    def init_pygame(self):
        """Initialize PyGame and OpenGL"""
        pygame.init()
        
        flags = DOUBLEBUF | OPENGL | RESIZABLE
        if self.fullscreen:
            flags |= FULLSCREEN
            
        self.screen = pygame.display.set_mode(self.display, flags)
        pygame.display.set_caption(self.title)
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        glClearColor(*self.colors['background'])
        glPointSize(5.0)
        glLineWidth(2.0)
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        
    def handle_common_events(self, event):
        """Handle common events for all viewers"""
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_F11:
                self.toggle_fullscreen()
            elif event.key == pygame.K_F1:
                self.show_ui = not self.show_ui
            elif event.key == pygame.K_F2:
                self.show_stats = not self.show_stats
            elif event.key == pygame.K_F5:
                self.change_theme()
            elif event.key == pygame.K_F12:
                self.take_screenshot()
        elif event.type == pygame.VIDEORESIZE:
            self.display = (event.w, event.h)
            glViewport(0, 0, event.w, event.h)
            
        return True
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        pygame.display.quit()
        pygame.display.init()
        self.init_pygame()
        
    def change_theme(self):
        """Cycle through themes"""
        themes = list(Theme)
        current_idx = themes.index(self.theme)
        self.theme = themes[(current_idx + 1) % len(themes)]
        self.colors = self.theme.get_colors()
        glClearColor(*self.colors['background'])
        
    def take_screenshot(self):
        """Save screenshot to file"""
        width, height = self.display
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        
        surface = pygame.image.fromstring(data, (width, height), 'RGBA')
        surface = pygame.transform.flip(surface, False, True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pygame.image.save(surface, filename)
        print(f"💾 Screenshot saved as {filename}")
        
    def draw_text(self, text, x, y, color=None, font=None, align_left=True):
        """Draw text using PyGame font"""
        if font is None:
            font = self.font
            
        color = color or self.colors['text']
        rgb_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        text_surface = font.render(text, True, rgb_color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        width, height = text_surface.get_size()
        
        if not align_left:
            x -= width
            
        glWindowPos2d(x, y)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
    def draw_2d_overlay(self):
        """Setup 2D overlay rendering"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.display[0], 0, self.display[1])
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for 2D overlay
        glDisable(GL_DEPTH_TEST)
        
    def end_2d_overlay(self):
        """End 2D overlay rendering"""
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
    def draw_fps(self):
        """Display FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
        self.draw_2d_overlay()
        self.draw_text(f"FPS: {self.fps:.1f}", 10, self.display[1] - 30)
        self.end_2d_overlay()
        
    def draw_progress_bar(self, x, y, width, height, progress, color=None):
        """Draw a progress bar"""
        color = color or self.colors['primary']
        
        self.draw_2d_overlay()
        
        # Background
        glColor4f(0.2, 0.2, 0.2, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        # Progress
        progress_width = max(0, min(width * progress, width))
        glColor4f(*color, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + progress_width, y)
        glVertex2f(x + progress_width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        # Border
        glColor4f(1, 1, 1, 1)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        self.end_2d_overlay()
        
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

# ============================================================================
# ENHANCED 2D VIEWER (UNIVERSAL)
# ============================================================================

class Interactive2DViewer(BaseViewer):
    """
    Enhanced 2D Trajectory Viewer with multiple view modes
    
    New Features:
    - Multiple simultaneous viewports
    - Real-time data streaming
    - Measurement tools
    - Export capabilities
    - Enhanced camera controls
    """
    
    def __init__(self, drones_data: Dict, conflicts: List = None):
        super().__init__("2D Trajectory Viewer - Enhanced", (1600, 1000))
        
        # Convert input data to proper objects
        self.drones = convert_to_drone_objects(drones_data)
        self.conflicts = convert_to_conflict_objects(conflicts or [])
        
        # View configuration
        self.view_modes = [ViewMode.XY, ViewMode.XZ, ViewMode.YZ, ViewMode.TIME_ALTITUDE]
        self.active_views = [0, 1, 2, 3]  # Show all views by default
        self.current_focus_view = 0
        self.split_view = True  # Show multiple views simultaneously
        
        # Camera per viewport
        self.viewport_cameras = [
            {'offset': [0, 0], 'zoom': 1.0} for _ in range(len(self.view_modes))
        ]
        
        # UI state
        self.show_conflicts = True
        self.show_grid = True
        self.show_labels = True
        self.show_measurement = False
        self.measurement_start = None
        self.measurement_end = None
        
        # Analysis tools
        self.selected_drone = None
        self.hovered_point = None
        self.time_cursor = 0.0
        self.max_time = self._calculate_max_time()
        
        # Colors
        self.drone_colors = self._generate_drone_colors()
        
        # Calculate bounds
        self._calculate_bounds()
        
        # Initialize
        self.init_pygame()
        
        self._print_controls()
        self.run()
        
    def _calculate_max_time(self):
        """Calculate maximum time from all drones"""
        max_t = 0
        for drone in self.drones.values():
            if drone.waypoints:
                max_t = max(max_t, max(wp.t for wp in drone.waypoints))
        return max_t
        
    def _calculate_bounds(self):
        """Calculate world bounds for all dimensions"""
        all_coords = {'x': [], 'y': [], 'z': [], 't': []}
        
        for drone in self.drones.values():
            for wp in drone.waypoints:
                all_coords['x'].append(wp.x)
                all_coords['y'].append(wp.y)
                all_coords['z'].append(wp.z)
                all_coords['t'].append(wp.t)
                
        self.bounds = {}
        for key, values in all_coords.items():
            if values:
                min_val, max_val = min(values), max(values)
                padding = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
                self.bounds[key] = (min_val - padding, max_val + padding)
            else:
                self.bounds[key] = (0, 1)
                
    def _generate_drone_colors(self):
        """Generate distinct colors for drones"""
        colors = []
        num_drones = len(self.drones)
        
        # Generate distinct colors using golden ratio
        golden_ratio_conjugate = 0.618033988749895
        
        for i in range(num_drones):
            hue = (i * golden_ratio_conjugate) % 1.0
            # Convert HSV to RGB
            h_i = int(hue * 6)
            f = hue * 6 - h_i
            q = 1 - f
            t = f
            
            if h_i == 0:
                r, g, b = 1, t, 0
            elif h_i == 1:
                r, g, b = q, 1, 0
            elif h_i == 2:
                r, g, b = 0, 1, t
            elif h_i == 3:
                r, g, b = 0, q, 1
            elif h_i == 4:
                r, g, b = t, 0, 1
            else:
                r, g, b = 1, 0, q
                
            colors.append((r, g, b))
            
        return dict(zip(self.drones.keys(), colors))
        
    def _print_controls(self):
        """Print control instructions"""
        controls = [
            "=" * 80,
            "ENHANCED 2D VIEWER CONTROLS",
            "=" * 80,
            "1/2/3/4    - Focus viewport (XY/XZ/YZ/Time-Alt)",
            "V          - Toggle split view / single view",
            "Mouse Drag - Pan view (in focused viewport)",
            "Wheel      - Zoom in/out",
            "SPACE      - Toggle conflict markers",
            "G          - Toggle grid",
            "M          - Toggle measurement tool",
            "TAB        - Select next drone",
            "←/→        - Move time cursor",
            "F5         - Change theme",
            "F11        - Toggle fullscreen",
            "F12        - Take screenshot",
            "S          - Export data",
            "ESC        - Exit",
            "=" * 80
        ]
        
        print("\n" + "\n".join(controls))
        
    def get_viewport_rect(self, view_idx):
        """Get rectangle for viewport"""
        if not self.split_view or view_idx not in self.active_views:
            # Full screen for focused view
            return pygame.Rect(0, 0, self.display[0], self.display[1])
            
        # Calculate grid layout
        num_views = len(self.active_views)
        if num_views <= 2:
            cols = 2
        elif num_views <= 4:
            cols = 2
        else:
            cols = 3
            
        rows = (num_views + cols - 1) // cols
        
        width = self.display[0] // cols
        height = self.display[1] // rows
        
        # Find position of this view
        idx_in_active = self.active_views.index(view_idx)
        row = idx_in_active // cols
        col = idx_in_active % cols
        
        margin = 2
        return pygame.Rect(
            col * width + margin,
            row * height + margin,
            width - 2 * margin,
            height - 2 * margin
        )
        
    def world_to_screen(self, x, y, view_idx):
        """Convert world coordinates to screen coordinates"""
        view_mode = self.view_modes[view_idx]
        camera = self.viewport_cameras[view_idx]
        
        # Get bounds for this view
        if view_mode == ViewMode.XY:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['y']
        elif view_mode == ViewMode.XZ:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['z']
        elif view_mode == ViewMode.YZ:
            x_bounds, y_bounds = self.bounds['y'], self.bounds['z']
        else:  # TIME_ALTITUDE
            x_bounds, y_bounds = self.bounds['t'], self.bounds['z']
            
        # Normalize
        x_norm = (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
        y_norm = (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
        
        # Apply camera transform
        x_screen = x_norm * camera['zoom'] + camera['offset'][0]
        y_screen = y_norm * camera['zoom'] + camera['offset'][1]
        
        return x_screen, y_screen
        
    def screen_to_world(self, x_screen, y_screen, view_idx):
        """Convert screen coordinates to world coordinates"""
        view_mode = self.view_modes[view_idx]
        camera = self.viewport_cameras[view_idx]
        
        # Get bounds
        if view_mode == ViewMode.XY:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['y']
        elif view_mode == ViewMode.XZ:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['z']
        elif view_mode == ViewMode.YZ:
            x_bounds, y_bounds = self.bounds['y'], self.bounds['z']
        else:  # TIME_ALTITUDE
            x_bounds, y_bounds = self.bounds['t'], self.bounds['z']
            
        # Remove camera transform
        x_norm = (x_screen - camera['offset'][0]) / camera['zoom']
        y_norm = (y_screen - camera['offset'][1]) / camera['zoom']
        
        # Convert to world
        x_world = x_bounds[0] + x_norm * (x_bounds[1] - x_bounds[0])
        y_world = y_bounds[0] + y_norm * (y_bounds[1] - y_bounds[0])
        
        return x_world, y_world
        
    def draw_viewport(self, view_idx):
        """Draw a single viewport"""
        viewport = self.get_viewport_rect(view_idx)
        
        # Set viewport
        glViewport(viewport.x, viewport.y, viewport.width, viewport.height)
        
        # Set projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Adjust for camera zoom and offset
        camera = self.viewport_cameras[view_idx]
        zoom = camera['zoom']
        offset_x, offset_y = camera['offset']
        
        left = -offset_x / zoom
        right = (viewport.width - offset_x) / zoom
        bottom = -(viewport.height - offset_y) / zoom
        top = offset_y / zoom
        
        glOrtho(left, right, bottom, top, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Draw grid
        if self.show_grid:
            self.draw_grid_viewport(view_idx)
            
        # Draw trajectories
        for drone_id, drone in self.drones.items():
            if drone.show_trajectory:
                self.draw_drone_trajectory(drone, view_idx)
                
        # Draw conflicts
        if self.show_conflicts:
            self.draw_conflicts_viewport(view_idx)
            
        # Draw measurement tool
        if self.show_measurement and self.measurement_start and self.measurement_end:
            self.draw_measurement_tool(view_idx)
            
        # Draw time cursor
        if view_idx == 3:  # Time-Altitude view
            self.draw_time_cursor(view_idx)
            
        # Draw viewport border
        glViewport(0, 0, self.display[0], self.display[1])
        self.draw_2d_overlay()
        
        glColor4f(1, 1, 1, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(viewport.x, viewport.y)
        glVertex2f(viewport.x + viewport.width, viewport.y)
        glVertex2f(viewport.x + viewport.width, viewport.y + viewport.height)
        glVertex2f(viewport.x, viewport.y + viewport.height)
        glEnd()
        
        # Draw viewport title
        title = self.view_modes[view_idx].value
        if view_idx == self.current_focus_view:
            title = f"★ {title}"
            
        self.draw_text(title, viewport.x + 10, viewport.y + viewport.height - 25)
        self.end_2d_overlay()
        
    def draw_grid_viewport(self, view_idx):
        """Draw grid for specific viewport"""
        view_mode = self.view_modes[view_idx]
        
        if view_mode == ViewMode.XY:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['y']
            x_label, y_label = "X", "Y"
        elif view_mode == ViewMode.XZ:
            x_bounds, y_bounds = self.bounds['x'], self.bounds['z']
            x_label, y_label = "X", "Z"
        elif view_mode == ViewMode.YZ:
            x_bounds, y_bounds = self.bounds['y'], self.bounds['z']
            x_label, y_label = "Y", "Z"
        else:  # TIME_ALTITUDE
            x_bounds, y_bounds = self.bounds['t'], self.bounds['z']
            x_label, y_label = "Time", "Altitude"
            
        glColor4f(*self.colors['grid'])
        glLineWidth(1.0)
        
        # Draw grid lines
        num_lines = 10
        for i in range(num_lines + 1):
            t = i / num_lines
            x = x_bounds[0] + t * (x_bounds[1] - x_bounds[0])
            y = y_bounds[0] + t * (y_bounds[1] - y_bounds[0])
            
            # Vertical lines
            glBegin(GL_LINES)
            glVertex2f(x, y_bounds[0])
            glVertex2f(x, y_bounds[1])
            glEnd()
            
            # Horizontal lines
            glBegin(GL_LINES)
            glVertex2f(x_bounds[0], y)
            glVertex2f(x_bounds[1], y)
            glEnd()
            
    def draw_drone_trajectory(self, drone, view_idx):
        """Draw drone trajectory in viewport"""
        if not drone.waypoints:
            return
            
        view_mode = self.view_modes[view_idx]
        color = drone.color if drone.is_primary else self.drone_colors.get(drone.id, (0.5, 0.5, 0.5))
        
        # Draw trajectory line
        glColor4f(*color, 0.7 if drone.is_primary else 0.4)
        glLineWidth(3.0 if drone.is_primary else 1.5)
        
        glBegin(GL_LINE_STRIP)
        for wp in drone.waypoints:
            if view_mode == ViewMode.XY:
                x, y = wp.x, wp.y
            elif view_mode == ViewMode.XZ:
                x, y = wp.x, wp.z
            elif view_mode == ViewMode.YZ:
                x, y = wp.y, wp.z
            else:  # TIME_ALTITUDE
                x, y = wp.t, wp.z
            glVertex2f(x, y)
        glEnd()
        
        # Draw waypoints
        glPointSize(8.0 if drone.is_primary else 4.0)
        glColor4f(*color, 1.0)
        
        glBegin(GL_POINTS)
        for wp in drone.waypoints:
            if view_mode == ViewMode.XY:
                x, y = wp.x, wp.y
            elif view_mode == ViewMode.XZ:
                x, y = wp.x, wp.z
            elif view_mode == ViewMode.YZ:
                x, y = wp.y, wp.z
            else:  # TIME_ALTITUDE
                x, y = wp.t, wp.z
            glVertex2f(x, y)
        glEnd()
        
        # Draw start and end markers for primary drone
        if drone.is_primary and len(drone.waypoints) >= 2:
            start_wp = drone.waypoints[0]
            end_wp = drone.waypoints[-1]
            
            if view_mode == ViewMode.XY:
                sx, sy = start_wp.x, start_wp.y
                ex, ey = end_wp.x, end_wp.y
            elif view_mode == ViewMode.XZ:
                sx, sy = start_wp.x, start_wp.z
                ex, ey = end_wp.x, end_wp.z
            elif view_mode == ViewMode.YZ:
                sx, sy = start_wp.y, start_wp.z
                ex, ey = end_wp.y, end_wp.z
            else:  # TIME_ALTITUDE
                sx, sy = start_wp.t, start_wp.z
                ex, ey = end_wp.t, end_wp.z
                
            # Start marker (green circle)
            self.draw_circle(sx, sy, 5, (0, 1, 0, 1))
            
            # End marker (red square)
            self.draw_square(ex, ey, 5, (1, 0, 0, 1))
            
    def draw_conflicts_viewport(self, view_idx):
        """Draw conflicts in viewport"""
        if not self.conflicts:
            return
            
        view_mode = self.view_modes[view_idx]
        severity_colors = {
            'CRITICAL': (1, 0, 0, 0.9),
            'HIGH': (1, 0.5, 0, 0.9),
            'MEDIUM': (1, 1, 0, 0.9),
            'LOW': (0, 1, 0, 0.9)
        }
        
        for conflict in self.conflicts:
            loc = conflict.location
            severity = conflict.severity
            
            if view_mode == ViewMode.XY:
                x, y = loc['x'], loc['y']
            elif view_mode == ViewMode.XZ:
                x, y = loc['x'], loc['z']
            elif view_mode == ViewMode.YZ:
                x, y = loc['y'], loc['z']
            else:  # TIME_ALTITUDE
                x, y = conflict.time, loc['z']
                
            color = severity_colors.get(severity, (0.5, 0.5, 0.5, 0.9))
            size = {'CRITICAL': 8, 'HIGH': 6, 'MEDIUM': 4, 'LOW': 2}.get(severity, 3)
            
            # Draw conflict marker (pulsing)
            pulse = 1.0 + 0.2 * math.sin(time.time() * 3)
            
            glColor4f(*color)
            glPointSize(size * pulse)
            
            glBegin(GL_POINTS)
            glVertex2f(x, y)
            glEnd()
            
    def draw_circle(self, x, y, radius, color):
        """Draw a filled circle"""
        glColor4f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(21):
            angle = 2 * math.pi * i / 20
            glVertex2f(x + radius * math.cos(angle), 
                      y + radius * math.sin(angle))
        glEnd()
        
    def draw_square(self, x, y, size, color):
        """Draw a filled square"""
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x - size, y - size)
        glVertex2f(x + size, y - size)
        glVertex2f(x + size, y + size)
        glVertex2f(x - size, y + size)
        glEnd()
        
    def draw_measurement_tool(self, view_idx):
        """Draw measurement line and distance"""
        if not self.measurement_start or not self.measurement_end:
            return
            
        start_world = self.screen_to_world(*self.measurement_start, view_idx)
        end_world = self.screen_to_world(*self.measurement_end, view_idx)
        
        # Calculate distance
        dx = end_world[0] - start_world[0]
        dy = end_world[1] - start_world[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Draw line
        glColor4f(1, 1, 0, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(start_world[0], start_world[1])
        glVertex2f(end_world[0], end_world[1])
        glEnd()
        
        # Draw endpoints
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex2f(start_world[0], start_world[1])
        glVertex2f(end_world[0], end_world[1])
        glEnd()
        
        # Draw distance text
        mid_x = (start_world[0] + end_world[0]) / 2
        mid_y = (start_world[1] + end_world[1]) / 2
        
        # Convert to screen for text
        mid_screen = self.world_to_screen(mid_x, mid_y, view_idx)
        viewport = self.get_viewport_rect(view_idx)
        
        self.draw_2d_overlay()
        self.draw_text(f"{distance:.1f}m", 
                      viewport.x + mid_screen[0], 
                      viewport.y + mid_screen[1],
                      color=(1, 1, 0, 1))
        self.end_2d_overlay()
        
    def draw_time_cursor(self, view_idx):
        """Draw vertical time cursor"""
        if view_idx != 3:  # Only for time-altitude view
            return
            
        glColor4f(1, 1, 1, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(self.time_cursor, self.bounds['z'][0])
        glVertex2f(self.time_cursor, self.bounds['z'][1])
        glEnd()
        
    def draw_ui(self):
        """Draw main UI overlay"""
        if not self.show_ui:
            return
            
        self.draw_2d_overlay()
        
        # Title
        self.draw_text("ENHANCED 2D TRAJECTORY VIEWER", 
                      self.display[0] // 2, self.display[1] - 30,
                      font=self.font_large, align_left=False)
        
        # Stats panel
        panel_x, panel_y = self.display[0] - 250, 50
        panel_width, panel_height = 230, 200
        
        # Panel background
        glColor4f(0, 0, 0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Panel border
        glColor4f(1, 1, 1, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Stats text
        text_y = panel_y + panel_height - 30
        stats = [
            f"Drones: {len(self.drones)}",
            f"Conflicts: {len(self.conflicts)}",
            f"Time: {self.time_cursor:.1f}s / {self.max_time:.1f}s",
            f"Views: {len(self.active_views)} active"
        ]
        
        for stat in stats:
            self.draw_text(stat, panel_x + 10, text_y)
            text_y -= 25
            
        # Control hints
        if self.show_stats:
            hints_y = 50
            hints = [
                "F1: Toggle UI",
                "F2: Toggle stats",
                "F5: Change theme",
                "V: Split/Single view",
                "M: Measurement tool"
            ]
            
            for hint in hints:
                self.draw_text(hint, 10, hints_y)
                hints_y += 25
                
        self.end_2d_overlay()
        
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if not self.handle_common_events(event):
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.current_focus_view = 0
                elif event.key == pygame.K_2:
                    self.current_focus_view = 1
                elif event.key == pygame.K_3:
                    self.current_focus_view = 2
                elif event.key == pygame.K_4:
                    self.current_focus_view = 3
                elif event.key == pygame.K_v:
                    self.split_view = not self.split_view
                elif event.key == pygame.K_SPACE:
                    self.show_conflicts = not self.show_conflicts
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_m:
                    self.show_measurement = not self.show_measurement
                    if not self.show_measurement:
                        self.measurement_start = None
                        self.measurement_end = None
                elif event.key == pygame.K_TAB:
                    drone_ids = list(self.drones.keys())
                    if drone_ids:
                        if self.selected_drone is None:
                            self.selected_drone = drone_ids[0]
                        else:
                            idx = drone_ids.index(self.selected_drone)
                            self.selected_drone = drone_ids[(idx + 1) % len(drone_ids)]
                elif event.key == pygame.K_LEFT:
                    self.time_cursor = max(0, self.time_cursor - 1.0)
                elif event.key == pygame.K_RIGHT:
                    self.time_cursor = min(self.max_time, self.time_cursor + 1.0)
                elif event.key == pygame.K_s:
                    self.export_data()
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.show_measurement:
                        mouse_pos = pygame.mouse.get_pos()
                        view_idx = self.get_viewport_at_pos(mouse_pos)
                        if view_idx == self.current_focus_view:
                            self.measurement_start = mouse_pos
                            self.measurement_end = None
                elif event.button == 3:  # Right click
                    # Select drone at position
                    mouse_pos = pygame.mouse.get_pos()
                    view_idx = self.get_viewport_at_pos(mouse_pos)
                    if view_idx is not None:
                        world_pos = self.screen_to_world(mouse_pos[0], mouse_pos[1], view_idx)
                        self.select_drone_at_position(world_pos, view_idx)
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.show_measurement and self.measurement_start:
                    self.measurement_end = pygame.mouse.get_pos()
                    
            elif event.type == pygame.MOUSEMOTION:
                # Update hovered point
                mouse_pos = pygame.mouse.get_pos()
                view_idx = self.get_viewport_at_pos(mouse_pos)
                if view_idx is not None:
                    self.hovered_point = self.screen_to_world(mouse_pos[0], mouse_pos[1], view_idx)
                    
        # Handle mouse wheel for zoom
        if pygame.mouse.get_pressed()[0]:  # Left drag for pan
            mouse_rel = pygame.mouse.get_rel()
            camera = self.viewport_cameras[self.current_focus_view]
            camera['offset'][0] += mouse_rel[0]
            camera['offset'][1] -= mouse_rel[1]  # Invert Y
        elif pygame.mouse.get_pressed()[2]:  # Right drag for zoom
            mouse_rel = pygame.mouse.get_rel()
            camera = self.viewport_cameras[self.current_focus_view]
            zoom_factor = 1.0 + mouse_rel[1] * 0.01
            camera['zoom'] *= zoom_factor
            
        return True
        
    def get_viewport_at_pos(self, pos):
        """Get viewport index at mouse position"""
        for view_idx in range(len(self.view_modes)):
            if view_idx in self.active_views:
                viewport = self.get_viewport_rect(view_idx)
                if viewport.collidepoint(pos):
                    return view_idx
        return None
        
    def select_drone_at_position(self, world_pos, view_idx):
        """Select drone closest to position"""
        view_mode = self.view_modes[view_idx]
        min_dist = float('inf')
        selected = None
        
        for drone_id, drone in self.drones.items():
            for wp in drone.waypoints:
                if view_mode == ViewMode.XY:
                    x, y = wp.x, wp.y
                elif view_mode == ViewMode.XZ:
                    x, y = wp.x, wp.z
                elif view_mode == ViewMode.YZ:
                    x, y = wp.y, wp.z
                else:  # TIME_ALTITUDE
                    x, y = wp.t, wp.z
                    
                dist = math.sqrt((x - world_pos[0])**2 + (y - world_pos[1])**2)
                if dist < min_dist and dist < 5:  # 5 unit threshold
                    min_dist = dist
                    selected = drone_id
                    
        if selected:
            self.selected_drone = selected
            print(f"Selected drone: {selected}")
            
    def export_data(self):
        """Export current view data"""
        data = {
            'drones': {d_id: {'waypoints': [(wp.x, wp.y, wp.z, wp.t) 
                                           for wp in d.waypoints]}
                      for d_id, d in self.drones.items()},
            'conflicts': [{'time': c.time,
                          'location': c.location,
                          'severity': c.severity,
                          'drones_involved': c.drones_involved}
                         for c in self.conflicts],
            'view_state': {
                'time_cursor': self.time_cursor,
                'active_views': self.active_views,
                'camera_states': self.viewport_cameras
            }
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"export_2d_view_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"📊 Data exported to {filename}")
        
    def render(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.split_view:
            # Draw all active viewports
            for view_idx in self.active_views:
                self.draw_viewport(view_idx)
        else:
            # Draw only focused viewport
            self.draw_viewport(self.current_focus_view)
            
        # Draw UI
        self.draw_ui()
        
        # Draw FPS
        self.draw_fps()
        
        pygame.display.flip()
        
    def run(self):
        """Main loop"""
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.update_fps()
            self.running = self.handle_events()
            self.render()
            
        pygame.quit()

# ============================================================================
# ENHANCED 3D VIEWER (UNIVERSAL)
# ============================================================================

class Interactive3DViewer(BaseViewer):
    """
    Enhanced 3D Trajectory Viewer
    
    New Features:
    - Multiple camera modes (orbit, FPS, free)
    - Drone model rendering
    - Enhanced lighting
    - Collision volume visualization
    - Path prediction
    """
    
    def __init__(self, drones_data: Dict, conflicts: List = None):
        super().__init__("3D Trajectory Viewer - Enhanced", (1600, 1000))
        
        # Convert input data to proper objects
        self.drones = convert_to_drone_objects(drones_data)
        self.conflicts = convert_to_conflict_objects(conflicts or [])
        
        # Camera modes
        self.camera_mode = "orbit"  # orbit, fps, free
        self.rotation_speed = 0.5
        self.move_speed = 10.0
        
        # Lighting
        self.light_enabled = True
        self.light_pos = [100, 100, 100, 1]
        self.ambient_light = [0.2, 0.2, 0.2, 1.0]
        self.diffuse_light = [0.8, 0.8, 0.8, 1.0]
        
        # Display options
        self.show_drone_models = True
        self.show_collision_volumes = True
        self.show_predicted_paths = True
        self.show_terrain = False
        
        # Animation
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.animation_paused = False
        
        # Selection
        self.selected_drone = None
        self.hovered_drone = None
        
        # Calculate bounds and setup
        self._calculate_bounds()
        self.init_pygame()
        self._setup_lighting()
        self._create_drone_model()
        
        self._print_controls()
        self.run()
        
    def _calculate_bounds(self):
        """Calculate 3D bounds"""
        all_points = []
        for drone in self.drones.values():
            for wp in drone.waypoints:
                all_points.append([wp.x, wp.y, wp.z])
                
        if all_points:
            all_points = np.array(all_points)
            self.center = np.mean(all_points, axis=0)
            self.max_range = np.max(np.ptp(all_points, axis=0)) * 0.6
        else:
            self.center = [0, 0, 0]
            self.max_range = 100
            
        # Set initial camera
        self.camera_target = self.center
        self.camera_position = [
            self.center[0],
            self.center[1] + self.max_range * 0.5,
            self.center[2] + self.max_range * 2
        ]
        
    def _setup_lighting(self):
        """Setup OpenGL lighting"""
        if self.light_enabled:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            glLightfv(GL_LIGHT0, GL_POSITION, self.light_pos)
            glLightfv(GL_LIGHT0, GL_AMBIENT, self.ambient_light)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, self.diffuse_light)
            glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
            
            glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
            glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        else:
            glDisable(GL_LIGHTING)
            
    def _create_drone_model(self):
        """Create display list for drone model"""
        try:
            self.drone_model = glGenLists(1)
            glNewList(self.drone_model, GL_COMPILE)
            
            # Draw drone as a simple quadcopter
            glPushMatrix()
            glScalef(2, 0.5, 2)
            
            # Body (central cube)
            glColor4f(0.3, 0.3, 0.3, 1.0)
            self.draw_cube(1.0)
            
            # Arms
            glColor4f(0.2, 0.2, 0.2, 1.0)
            arm_length = 1.5
            arm_thickness = 0.1
            
            # Front arm
            glPushMatrix()
            glTranslatef(0, 0, arm_length/2)
            glScalef(arm_thickness, arm_thickness, arm_length)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Back arm
            glPushMatrix()
            glTranslatef(0, 0, -arm_length/2)
            glScalef(arm_thickness, arm_thickness, arm_length)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Right arm
            glPushMatrix()
            glTranslatef(arm_length/2, 0, 0)
            glScalef(arm_length, arm_thickness, arm_thickness)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Left arm
            glPushMatrix()
            glTranslatef(-arm_length/2, 0, 0)
            glScalef(arm_length, arm_thickness, arm_thickness)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Propellers (disks at ends)
            glColor4f(0.8, 0.8, 0.8, 1.0)
            
            positions = [
                (arm_length/2, 0, 0),
                (-arm_length/2, 0, 0),
                (0, 0, arm_length/2),
                (0, 0, -arm_length/2)
            ]
            
            for pos in positions:
                glPushMatrix()
                glTranslatef(*pos)
                glRotatef(90, 1, 0, 0)  # Make it horizontal
                self.draw_torus(0.05, 0.4, 8, 16)
                glPopMatrix()
                
            glPopMatrix()
            glEndList()
        except:
            # Fallback if GLUT not available
            self.drone_model = None
            
    def draw_cube(self, size):
        """Draw a cube (fallback if glutSolidCube not available)"""
        try:
            glutSolidCube(size)
        except:
            # Manual cube drawing
            s = size / 2
            glBegin(GL_QUADS)
            # Front
            glVertex3f(-s, -s, s)
            glVertex3f(s, -s, s)
            glVertex3f(s, s, s)
            glVertex3f(-s, s, s)
            # Back
            glVertex3f(-s, -s, -s)
            glVertex3f(-s, s, -s)
            glVertex3f(s, s, -s)
            glVertex3f(s, -s, -s)
            # Top
            glVertex3f(-s, s, -s)
            glVertex3f(-s, s, s)
            glVertex3f(s, s, s)
            glVertex3f(s, s, -s)
            # Bottom
            glVertex3f(-s, -s, -s)
            glVertex3f(s, -s, -s)
            glVertex3f(s, -s, s)
            glVertex3f(-s, -s, s)
            # Right
            glVertex3f(s, -s, -s)
            glVertex3f(s, s, -s)
            glVertex3f(s, s, s)
            glVertex3f(s, -s, s)
            # Left
            glVertex3f(-s, -s, -s)
            glVertex3f(-s, -s, s)
            glVertex3f(-s, s, s)
            glVertex3f(-s, s, -s)
            glEnd()
            
    def draw_torus(self, inner_radius, outer_radius, sides, rings):
        """Draw a torus (fallback if glutSolidTorus not available)"""
        try:
            glutSolidTorus(inner_radius, outer_radius, sides, rings)
        except:
            # Simple circle as fallback
            glBegin(GL_TRIANGLE_FAN)
            for i in range(32):
                angle = 2 * math.pi * i / 32
                glVertex3f(outer_radius * math.cos(angle), 
                          outer_radius * math.sin(angle), 
                          0)
            glEnd()
        
    def _print_controls(self):
        """Print control instructions"""
        controls = [
            "=" * 80,
            "ENHANCED 3D VIEWER CONTROLS",
            "=" * 80,
            "Mouse Drag    - Rotate view",
            "WASD          - Move camera",
            "QE            - Move up/down",
            "Wheel         - Zoom in/out",
            "C             - Cycle camera modes",
            "L             - Toggle lighting",
            "D             - Toggle drone models",
            "V             - Toggle collision volumes",
            "P             - Toggle predicted paths",
            "SPACE         - Play/pause animation",
            "+/-           - Adjust animation speed",
            "F             - Follow selected drone",
            "R             - Reset view",
            "F5            - Change theme",
            "F11           - Toggle fullscreen",
            "F12           - Take screenshot",
            "ESC           - Exit",
            "=" * 80
        ]
        
        print("\n" + "\n".join(controls))
        
    def setup_camera(self):
        """Setup camera based on current mode"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.display[0]/self.display[1], 0.1, 10000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if self.camera_mode == "orbit":
            gluLookAt(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                     self.camera_target[0], self.camera_target[1], self.camera_target[2],
                     0, 1, 0)
        else:
            # Calculate forward vector
            forward = np.array(self.camera_target) - np.array(self.camera_position)
            forward = forward / np.linalg.norm(forward)
            
            # Calculate right vector
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Recalculate up
            up = np.cross(right, forward)
            
            target = self.camera_position + forward
            
            gluLookAt(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                     target[0], target[1], target[2],
                     up[0], up[1], up[2])
                     
    def draw_coordinate_system(self):
        """Draw XYZ coordinate system"""
        glDisable(GL_LIGHTING)
        length = self.max_range * 0.2
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X - Red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        
        # Y - Green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        
        # Z - Blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        
        glEnd()
        glEnable(GL_LIGHTING)
        
    def draw_grid(self):
        """Draw ground grid"""
        glDisable(GL_LIGHTING)
        glColor4f(*self.colors['grid'])
        glLineWidth(1.0)
        
        size = self.max_range * 2
        divisions = 20
        step = size / divisions
        
        glBegin(GL_LINES)
        for i in range(-divisions, divisions + 1):
            pos = i * step
            
            # X lines
            glVertex3f(-size, 0, pos)
            glVertex3f(size, 0, pos)
            
            # Z lines
            glVertex3f(pos, 0, -size)
            glVertex3f(pos, 0, size)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def draw_drone(self, drone, position, rotation=0.0):
        """Draw a drone at position"""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glRotatef(rotation, 0, 1, 0)  # Rotate around Y axis
        
        color = drone.color if drone.is_primary else self.colors['primary']
        glColor4f(*color)
        
        if self.show_drone_models and self.drone_model:
            glCallList(self.drone_model)
        else:
            # Draw simple sphere
            self.draw_sphere(drone.radius, 16, 16)
            
        # Draw collision volume
        if self.show_collision_volumes:
            glDisable(GL_LIGHTING)
            glColor4f(*color, 0.2)
            self.draw_wire_sphere(drone.radius * 3, 8, 8)  # 3x radius for safety volume
            glEnable(GL_LIGHTING)
            
        glPopMatrix()
        
    def draw_sphere(self, radius, slices, stacks):
        """Draw a sphere"""
        try:
            quad = gluNewQuadric()
            gluSphere(quad, radius, slices, stacks)
            gluDeleteQuadric(quad)
        except:
            # Fallback: simple sphere approximation
            glBegin(GL_TRIANGLE_FAN)
            for i in range(slices + 1):
                lat0 = math.pi * (-0.5 + float(i - 1) / slices)
                z0 = radius * math.sin(lat0)
                zr0 = radius * math.cos(lat0)
                
                lat1 = math.pi * (-0.5 + float(i) / slices)
                z1 = radius * math.sin(lat1)
                zr1 = radius * math.cos(lat1)
                
                for j in range(stacks + 1):
                    lng = 2 * math.pi * float(j - 1) / stacks
                    x = math.cos(lng)
                    y = math.sin(lng)
                    
                    glNormal3f(x * zr0, y * zr0, z0)
                    glVertex3f(x * zr0, y * zr0, z0)
                    glNormal3f(x * zr1, y * zr1, z1)
                    glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
            
    def draw_wire_sphere(self, radius, slices, stacks):
        """Draw a wireframe sphere"""
        glBegin(GL_LINES)
        # Draw longitude lines
        for i in range(slices):
            lat0 = math.pi * (-0.5 + float(i) / slices)
            lat1 = math.pi * (-0.5 + float(i + 1) / slices)
            
            for j in range(stacks):
                lng = 2 * math.pi * float(j) / stacks
                x0 = radius * math.cos(lat0) * math.cos(lng)
                y0 = radius * math.cos(lat0) * math.sin(lng)
                z0 = radius * math.sin(lat0)
                
                x1 = radius * math.cos(lat1) * math.cos(lng)
                y1 = radius * math.cos(lat1) * math.sin(lng)
                z1 = radius * math.sin(lat1)
                
                glVertex3f(x0, y0, z0)
                glVertex3f(x1, y1, z1)
                
        # Draw latitude lines
        for i in range(stacks):
            lng0 = 2 * math.pi * float(i) / stacks
            lng1 = 2 * math.pi * float(i + 1) / stacks
            
            for j in range(slices):
                lat = math.pi * (-0.5 + float(j) / slices)
                x0 = radius * math.cos(lat) * math.cos(lng0)
                y0 = radius * math.cos(lat) * math.sin(lng0)
                z0 = radius * math.sin(lat)
                
                x1 = radius * math.cos(lat) * math.cos(lng1)
                y1 = radius * math.cos(lat) * math.sin(lng1)
                z1 = radius * math.sin(lat)
                
                glVertex3f(x0, y0, z0)
                glVertex3f(x1, y1, z1)
        glEnd()
        
    def draw_trajectory(self, drone):
        """Draw drone trajectory"""
        if not drone.waypoints or not drone.show_trajectory:
            return
            
        glDisable(GL_LIGHTING)
        color = drone.color if drone.is_primary else (*self.colors['secondary'], 0.5)
        glColor4f(*color)
        glLineWidth(2.0 if drone.is_primary else 1.0)
        
        glBegin(GL_LINE_STRIP)
        for wp in drone.waypoints:
            glVertex3f(wp.x, wp.y, wp.z)
        glEnd()
        
        # Draw waypoints
        glPointSize(6.0 if drone.is_primary else 3.0)
        glBegin(GL_POINTS)
        for wp in drone.waypoints:
            glVertex3f(wp.x, wp.y, wp.z)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def draw_predicted_path(self, drone):
        """Draw predicted future path"""
        if not self.show_predicted_paths or not drone.waypoints:
            return
            
        # Simple prediction: continue with current velocity
        if len(drone.waypoints) >= 2:
            last_wp = drone.waypoints[-1]
            second_last = drone.waypoints[-2]
            
            dt = last_wp.t - second_last.t
            if dt > 0:
                vx = (last_wp.x - second_last.x) / dt
                vy = (last_wp.y - second_last.y) / dt
                vz = (last_wp.z - second_last.z) / dt
                
                prediction_time = 5.0  # Predict 5 seconds ahead
                steps = 10
                
                glDisable(GL_LIGHTING)
                glColor4f(1, 1, 0, 0.5)
                glLineWidth(1.0)
                
                glBegin(GL_LINE_STRIP)
                glVertex3f(last_wp.x, last_wp.y, last_wp.z)
                
                for i in range(1, steps + 1):
                    t = i * prediction_time / steps
                    x = last_wp.x + vx * t
                    y = last_wp.y + vy * t
                    z = last_wp.z + vz * t
                    glVertex3f(x, y, z)
                    
                glEnd()
                glEnable(GL_LIGHTING)
                
    def draw_conflicts(self):
        """Draw conflict zones"""
        if not self.conflicts:
            return
            
        glDisable(GL_LIGHTING)
        
        severity_colors = {
            'CRITICAL': (1, 0, 0, 0.3),
            'HIGH': (1, 0.5, 0, 0.3),
            'MEDIUM': (1, 1, 0, 0.3),
            'LOW': (0, 1, 0, 0.3)
        }
        
        for conflict in self.conflicts:
            loc = conflict.location
            severity = conflict.severity
            color = severity_colors.get(severity, (0.5, 0.5, 0.5, 0.3))
            
            # Draw sphere for conflict zone
            radius = conflict.distance * 1.5
            
            glColor4f(*color)
            
            # Wireframe sphere
            glEnable(GL_BLEND)
            glDepthMask(GL_FALSE)
            self.draw_wire_sphere(radius, 8, 8)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)
            
        glEnable(GL_LIGHTING)
        
    def get_drone_position_at_time(self, drone, t):
        """Get interpolated drone position at time t"""
        if not drone.waypoints:
            return None
            
        # Find segment containing time t
        for i in range(len(drone.waypoints) - 1):
            wp1, wp2 = drone.waypoints[i], drone.waypoints[i + 1]
            if wp1.t <= t <= wp2.t:
                # Linear interpolation
                alpha = (t - wp1.t) / (wp2.t - wp1.t)
                x = wp1.x + alpha * (wp2.x - wp1.x)
                y = wp1.y + alpha * (wp2.y - wp1.y)
                z = wp1.z + alpha * (wp2.z - wp1.z)
                return np.array([x, y, z])
                
        return None
        
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if not self.handle_common_events(event):
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    modes = ["orbit", "fps", "free"]
                    current_idx = modes.index(self.camera_mode)
                    self.camera_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"Camera mode: {self.camera_mode}")
                elif event.key == pygame.K_l:
                    self.light_enabled = not self.light_enabled
                    if self.light_enabled:
                        glEnable(GL_LIGHTING)
                    else:
                        glDisable(GL_LIGHTING)
                elif event.key == pygame.K_d:
                    self.show_drone_models = not self.show_drone_models
                elif event.key == pygame.K_v:
                    self.show_collision_volumes = not self.show_collision_volumes
                elif event.key == pygame.K_p:
                    self.show_predicted_paths = not self.show_predicted_paths
                elif event.key == pygame.K_SPACE:
                    self.animation_paused = not self.animation_paused
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.animation_speed *= 1.5
                elif event.key == pygame.K_MINUS:
                    self.animation_speed /= 1.5
                elif event.key == pygame.K_f:
                    if self.selected_drone:
                        drone = self.drones[self.selected_drone]
                        pos = self.get_drone_position_at_time(drone, self.animation_time)
                        if pos is not None:
                            self.camera_target = pos
                            self.camera_position = pos + np.array([-20, 10, -20])
                elif event.key == pygame.K_r:
                    self._calculate_bounds()  # Reset camera
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check for drone selection
                    mouse_pos = pygame.mouse.get_pos()
                    self.select_drone_at_screen(mouse_pos)
                    
        # Handle continuous key presses for movement
        keys = pygame.key.get_pressed()
        dt = self.clock.get_time() / 1000.0
        
        # Camera movement based on mode
        if self.camera_mode in ["fps", "free"]:
            forward = np.array(self.camera_target) - np.array(self.camera_position)
            forward = forward / np.linalg.norm(forward)
            
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            move_speed = self.move_speed * dt
            
            if keys[pygame.K_w]:
                self.camera_position += forward * move_speed
                self.camera_target += forward * move_speed
            if keys[pygame.K_s]:
                self.camera_position -= forward * move_speed
                self.camera_target -= forward * move_speed
            if keys[pygame.K_a]:
                self.camera_position -= right * move_speed
                self.camera_target -= right * move_speed
            if keys[pygame.K_d]:
                self.camera_position += right * move_speed
                self.camera_target += right * move_speed
            if keys[pygame.K_q]:
                self.camera_position[1] -= move_speed
                self.camera_target[1] -= move_speed
            if keys[pygame.K_e]:
                self.camera_position[1] += move_speed
                self.camera_target[1] += move_speed
                
        # Mouse look for FPS mode
        if self.camera_mode == "fps" and pygame.mouse.get_pressed()[0]:
            dx, dy = pygame.mouse.get_rel()
            sensitivity = 0.1
            
            # Calculate rotation
            forward = np.array(self.camera_target) - np.array(self.camera_position)
            length = np.linalg.norm(forward)
            
            # Horizontal rotation (yaw)
            yaw = np.arctan2(forward[2], forward[0])
            yaw += dx * sensitivity
            
            # Vertical rotation (pitch)
            pitch = np.arcsin(forward[1] / length)
            pitch -= dy * sensitivity
            pitch = np.clip(pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)
            
            # Calculate new forward vector
            new_forward = np.array([
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.sin(yaw) * np.cos(pitch)
            ])
            
            self.camera_target = self.camera_position + new_forward * length
            
        return True
        
    def select_drone_at_screen(self, screen_pos):
        """Select drone by clicking (simple ray casting)"""
        # Simple implementation - just check all drones
        drone_ids = list(self.drones.keys())
        if drone_ids:
            if self.selected_drone is None:
                self.selected_drone = drone_ids[0]
            else:
                idx = drone_ids.index(self.selected_drone)
                self.selected_drone = drone_ids[(idx + 1) % len(drone_ids)]
                
            print(f"Selected drone: {self.selected_drone}")
            
    def update_animation(self, dt):
        """Update animation time"""
        if not self.animation_paused:
            max_time = max((max(wp.t for wp in drone.waypoints) 
                          for drone in self.drones.values() if drone.waypoints), 
                         default=0)
            self.animation_time += dt * self.animation_speed
            if self.animation_time > max_time:
                self.animation_time = 0
                
    def render(self):
        """Render the 3D scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        self.setup_camera()
        
        # Setup lighting
        if self.light_enabled:
            glLightfv(GL_LIGHT0, GL_POSITION, self.light_pos)
            
        # Draw environment
        self.draw_grid()
        self.draw_coordinate_system()
        
        # Draw conflicts
        self.draw_conflicts()
        
        # Draw all drones and their trajectories
        for drone_id, drone in self.drones.items():
            # Draw trajectory
            self.draw_trajectory(drone)
            
            # Draw predicted path
            self.draw_predicted_path(drone)
            
            # Draw drone at current animation time
            position = self.get_drone_position_at_time(drone, self.animation_time)
            if position is not None:
                # Calculate heading from velocity if available
                rotation = 0
                if len(drone.waypoints) >= 2:
                    # Find current segment
                    for i in range(len(drone.waypoints) - 1):
                        wp1, wp2 = drone.waypoints[i], drone.waypoints[i + 1]
                        if wp1.t <= self.animation_time <= wp2.t:
                            dx = wp2.x - wp1.x
                            dz = wp2.z - wp1.z
                            if dx != 0 or dz != 0:
                                rotation = np.degrees(np.arctan2(dz, dx))
                            break
                            
                self.draw_drone(drone, position, rotation)
                
        # Draw UI
        self.draw_ui()
        
        # Draw FPS
        self.draw_fps()
        
        pygame.display.flip()
        
    def draw_ui(self):
        """Draw 3D viewer UI"""
        if not self.show_ui:
            return
            
        self.draw_2d_overlay()
        
        # Title
        self.draw_text("ENHANCED 3D TRAJECTORY VIEWER", 
                      self.display[0] // 2, self.display[1] - 30,
                      font=self.font_large, align_left=False)
        
        # Info panel
        panel_x, panel_y = 20, self.display[1] - 150
        panel_width, panel_height = 300, 130
        
        # Panel background
        glColor4f(0, 0, 0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Panel border
        glColor4f(1, 1, 1, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Info text
        text_y = panel_y + panel_height - 30
        info_lines = [
            f"Time: {self.animation_time:.1f}s",
            f"Speed: {self.animation_speed:.1f}x",
            f"Camera: {self.camera_mode}",
            f"Drones: {len(self.drones)}",
            f"Selected: {self.selected_drone or 'None'}"
        ]
        
        for line in info_lines:
            self.draw_text(line, panel_x + 10, text_y)
            text_y -= 25
            
        # Status indicators
        status_x = panel_x + panel_width + 20
        status_y = panel_y + panel_height - 30
        
        status_items = [
            ("Models", self.show_drone_models),
            ("Volumes", self.show_collision_volumes),
            ("Predict", self.show_predicted_paths),
            ("Light", self.light_enabled),
            ("Paused", self.animation_paused)
        ]
        
        for label, state in status_items:
            color = (0, 1, 0) if state else (1, 0, 0)
            self.draw_text(f"{label}: {'ON' if state else 'OFF'}", 
                          status_x, status_y, color=color)
            status_y -= 25
            
        self.end_2d_overlay()
        
    def run(self):
        """Main loop"""
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.update_fps()
            self.update_animation(dt)
            self.running = self.handle_events()
            self.render()
            
        pygame.quit()

# ============================================================================
# SIMPLIFIED COMPATIBILITY WRAPPERS
# ============================================================================

class Simple2DViewer:
    """
    Simple 2D viewer for backward compatibility
    """
    
    def __init__(self, drones_data: Dict, conflicts: List = None):
        # Just use the enhanced viewer directly
        self.viewer = Interactive2DViewer(drones_data, conflicts)
        
class Simple3DViewer:
    """
    Simple 3D viewer for backward compatibility
    """
    
    def __init__(self, drones_data: Dict, conflicts: List = None):
        # Just use the enhanced viewer directly
        self.viewer = Interactive3DViewer(drones_data, conflicts)

# ============================================================================
# MAIN LAUNCHER FUNCTION
# ============================================================================

def launch_interactive_viewer(drones_data: Dict, conflicts: List = None, viewer_type='menu'):
    """
    Main function to launch interactive viewer
    
    Args:
        drones_data: Dictionary of drone data (can be in any format)
        conflicts: List of conflicts (can be dicts or objects)
        viewer_type: '2d', '3d', '4d', or 'menu' to choose
    """
    
    if not OPENGL_AVAILABLE:
        print("❌ OpenGL/PyGame not available!")
        return
    
    print(f"🎮 Launching interactive visualization...")
    print(f"   Drones: {len(drones_data)}")
    print(f"   Conflicts: {len(conflicts) if conflicts else 0}")
    
    if viewer_type == 'menu':
        print("\n" + "="*80)
        print("INTERACTIVE VISUALIZATION MENU")
        print("="*80)
        print("Choose viewer type:")
        print("  1. 2D Multi-View (Enhanced)")
        print("  2. 3D Trajectory Viewer (Enhanced)")
        print("  3. Launch Both (2D + 3D)")
        print("="*80)
        
        try:
            choice = input("Enter choice (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
            
        if choice == '1':
            viewer_type = '2d'
        elif choice == '2':
            viewer_type = '3d'
        elif choice == '3':
            # Launch both sequentially
            print("\n🚀 Launching 2D viewer...")
            Interactive2DViewer(drones_data, conflicts)
            print("\n🚀 Launching 3D viewer...")
            Interactive3DViewer(drones_data, conflicts)
            return
        else:
            print("⚠️  Invalid choice. Using 2D viewer.")
            viewer_type = '2d'
    
    if viewer_type == '2d':
        viewer = Interactive2DViewer(drones_data, conflicts)
    elif viewer_type == '3d':
        viewer = Interactive3DViewer(drones_data, conflicts)
    elif viewer_type == '4d':
        # For backward compatibility, use 3D viewer
        print("⚠️  4D viewer not available. Using 3D viewer instead.")
        viewer = Interactive3DViewer(drones_data, conflicts)
    else:
        print(f"⚠️  Unknown viewer type: {viewer_type}. Using 2D viewer.")
        viewer = Interactive2DViewer(drones_data, conflicts)

# ============================================================================
# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

def launch_2d_viewer(drones_data: Dict, conflicts: List = None):
    """Legacy function for backward compatibility"""
    launch_interactive_viewer(drones_data, conflicts, '2d')

def launch_3d_viewer(drones_data: Dict, conflicts: List = None):
    """Legacy function for backward compatibility"""
    launch_interactive_viewer(drones_data, conflicts, '3d')

def launch_4d_viewer(drones_data: Dict, conflicts: List = None):
    """Legacy function for backward compatibility"""
    launch_interactive_viewer(drones_data, conflicts, '4d')

# ============================================================================
# DIRECT SIMPLE LAUNCHERS
# ============================================================================

def launch_2d_simple(drones_data: Dict, conflicts: List = None):
    """Simple 2D viewer launcher"""
    Interactive2DViewer(drones_data, conflicts)

def launch_3d_simple(drones_data: Dict, conflicts: List = None):
    """Simple 3D viewer launcher"""
    Interactive3DViewer(drones_data, conflicts)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("UNIVERSAL INTERACTIVE VISUALIZATION SYSTEM")
    print("Handles any data format: dicts, objects, tuples, etc.")
    print("="*80)
    
    # Test with simulated Waypoint4D-like objects
    class Waypoint4D:
        def __init__(self, x, y, z, t):
            self.x = x
            self.y = y
            self.z = z
            self.t = t
            self.vx = 0
            self.vy = 0
            self.vz = 0
            
    # Create sample data in various formats
    drones = {}
    
    # Format 1: Waypoint4D objects
    drones['drone_001'] = {
        'waypoints': [
            Waypoint4D(0, 0, 10, 0),
            Waypoint4D(10, 0, 20, 10),
            Waypoint4D(20, 10, 30, 20),
            Waypoint4D(30, 20, 40, 30)
        ],
        'is_primary': True,
        'color': (1, 0, 0, 1)
    }
    
    # Format 2: Dictionaries
    drones['drone_002'] = {
        'waypoints': [
            {'x': 5, 'y': 5, 'z': 15, 't': 0},
            {'x': 15, 'y': 5, 'z': 25, 't': 10},
            {'x': 25, 'y': 15, 'z': 35, 't': 20},
            {'x': 35, 'y': 25, 'z': 45, 't': 30}
        ],
        'is_primary': False,
        'color': (0, 0.5, 1, 1)
    }
    
    # Format 3: Tuples
    drones['drone_003'] = {
        'waypoints': [
            (10, 0, 10, 0),
            (20, 0, 20, 10),
            (30, 10, 30, 20),
            (40, 20, 40, 30)
        ],
        'is_primary': False,
        'color': (0.5, 1, 0, 1)
    }
    
    # Sample conflicts
    conflicts = [
        {
            'time': 15,
            'location': Waypoint4D(25, 10, 25, 15),  # Object format
            'severity': 'HIGH',
            'drones_involved': ['drone_001', 'drone_002'],
            'distance': 5.0
        },
        {
            'time': 25,
            'location': {'x': 35, 'y': 20, 'z': 35},  # Dict format
            'severity': 'MEDIUM',
            'drones_involved': ['drone_002', 'drone_003'],
            'distance': 8.0
        }
    ]
    
    print("\n1. Test with mixed data formats")
    print("2. Exit")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = '2'
        
    if choice == '1':
        print("\n🚀 Testing universal data conversion...")
        launch_interactive_viewer(drones, conflicts, 'menu')
    else:
        print("Exiting...")