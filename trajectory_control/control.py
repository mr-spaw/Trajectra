"""
ULTRA-OPTIMIZED DRONE CONFLICT DETECTION SYSTEM
Heavy Multithreading + Parallel Computing + Advanced 4D Visualization
"""

# CRITICAL: Set matplotlib to non-interactive backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# OpenGL for 4D visualization
try:
    import pygame
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    print("Warning: OpenGL/pygame not available. 4D visualization disabled.")
    OPENGL_AVAILABLE = False

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
import itertools
from collections import defaultdict, deque
import math
import threading
from queue import Queue
import gc

# Try to use numba for JIT compilation
try:
    from numba import jit, prange, njit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not available. Using standard Python (slower).")
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    njit = jit
    prange = range

# ============================================================================
# 1. CORE DATA STRUCTURES (Optimized with slots)
# ============================================================================

class ConflictSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Waypoint4D:
    """4D waypoint with state - optimized"""
    t: float
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.t], dtype=np.float32)

@dataclass
class TrajectorySegment:
    """Continuous 4D trajectory segment - optimized"""
    
    id: int
    drone_id: str
    segment_idx: int
    start: Waypoint4D
    end: Waypoint4D
    safety_radius: float = 5.0
    _velocity: np.ndarray = field(init=False, repr=False)
    _duration: float = field(init=False, repr=False)
    _length: float = field(init=False, repr=False)
    _aabb_cache: Tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)
    
    def __post_init__(self):
        self._duration = self.end.t - self.start.t
        self._length = np.sqrt(
            (self.end.x - self.start.x)**2 +
            (self.end.y - self.start.y)**2 +
            (self.end.z - self.start.z)**2
        )
        
        if self._duration > 0:
            self._velocity = np.array([
                (self.end.x - self.start.x) / self._duration,
                (self.end.y - self.start.y) / self._duration,
                (self.end.z - self.start.z) / self._duration
            ], dtype=np.float32)
        else:
            self._velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Precompute AABB
        self._aabb_cache = self._compute_aabb()
    
    def _compute_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute Axis-Aligned Bounding Box"""
        min_vals = np.array([
            min(self.start.x, self.end.x) - self.safety_radius,
            min(self.start.y, self.end.y) - self.safety_radius,
            min(self.start.z, self.end.z) - self.safety_radius,
            self.start.t
        ], dtype=np.float32)
        
        max_vals = np.array([
            max(self.start.x, self.end.x) + self.safety_radius,
            max(self.start.y, self.end.y) + self.safety_radius,
            max(self.start.z, self.end.z) + self.safety_radius,
            self.end.t
        ], dtype=np.float32)
        
        return min_vals, max_vals
    
    @property
    def velocity(self) -> np.ndarray:
        return self._velocity
    
    @property
    def duration(self) -> float:
        return self._duration
    
    @property 
    def length(self) -> float:
        return self._length
    
    @property
    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._aabb_cache
    
    def position_at(self, t: float) -> Optional[np.ndarray]:
        if t < self.start.t or t > self.end.t:
            return None
        
        if self._duration == 0:
            return np.array([self.start.x, self.start.y, self.start.z], dtype=np.float32)
        
        alpha = (t - self.start.t) / self._duration
        alpha = max(0.0, min(1.0, alpha))
        
        return np.array([
            self.start.x + alpha * (self.end.x - self.start.x),
            self.start.y + alpha * (self.end.y - self.start.y),
            self.start.z + alpha * (self.end.z - self.start.z)
        ], dtype=np.float32)

# ============================================================================
# 2. OPTIMIZED ALGORITHMS (Vectorized + JIT)
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def fast_norm(vec):
        """Fast L2 norm"""
        return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    
    @njit(fastmath=True, cache=True)
    def fast_dot(v1, v2):
        """Fast dot product"""
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
else:
    def fast_norm(vec):
        return np.linalg.norm(vec)
    
    def fast_dot(v1, v2):
        return np.dot(v1, v2)

def aabb_intersect(aabb1, aabb2):
    """Fast AABB intersection test"""
    min1, max1 = aabb1
    min2, max2 = aabb2
    return np.all(min1 <= max2) and np.all(min2 <= max1)

def continuous_closest_approach_optimized(seg1, seg2):
    """Optimized 4D Continuous Closest Point of Approach"""
    # Early AABB rejection
    if not aabb_intersect(seg1.aabb, seg2.aabb):
        return float('inf'), 0.0, None
    
    t_start = max(seg1.start.t, seg2.start.t)
    t_end = min(seg1.end.t, seg2.end.t)
    
    if t_start > t_end:
        return float('inf'), t_start, None
    
    # Use precomputed velocities
    v_rel = seg2.velocity - seg1.velocity
    
    p1_start = np.array([seg1.start.x, seg1.start.y, seg1.start.z], dtype=np.float32)
    p2_start = np.array([seg2.start.x, seg2.start.y, seg2.start.z], dtype=np.float32)
    r0 = p2_start - p1_start
    
    v_dot = fast_dot(v_rel, v_rel)
    if v_dot > 1e-10:
        t_ca = -fast_dot(v_rel, r0) / v_dot
        t_ca = seg1.start.t + t_ca
        t_ca = max(t_start, min(t_end, t_ca))
    else:
        t_ca = t_start
    
    pos1 = seg1.position_at(t_ca)
    pos2 = seg2.position_at(t_ca)
    
    if pos1 is None or pos2 is None:
        return float('inf'), t_ca, None
    
    distance = fast_norm(pos2 - pos1)
    collision_point = (pos1 + pos2) * 0.5
    
    return float(distance), float(t_ca), collision_point

def batch_conflict_detection(primary_segments, other_segments, safety_radius):
    """Vectorized batch conflict detection"""
    conflicts = []
    
    for seg1 in primary_segments:
        for seg2 in other_segments:
            distance, t_ca, collision_point = continuous_closest_approach_optimized(seg1, seg2)
            
            if distance < safety_radius * 2:
                pos1 = seg1.position_at(t_ca)
                pos2 = seg2.position_at(t_ca)
                
                if pos1 is None or pos2 is None:
                    continue
                
                # Determine severity
                if distance < 0.3 * safety_radius:
                    severity = "CRITICAL"
                elif distance < 0.6 * safety_radius:
                    severity = "HIGH"
                elif distance < safety_radius:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                conflict = {
                    'primary_drone': seg1.drone_id,
                    'conflicting_drone': seg2.drone_id,
                    'time': float(t_ca),
                    'distance': float(distance),
                    'severity': severity,
                    'location': {
                        'x': float(collision_point[0]),
                        'y': float(collision_point[1]),
                        'z': float(collision_point[2])
                    },
                    'primary_location': {
                        'x': float(pos1[0]),
                        'y': float(pos1[1]),
                        'z': float(pos1[2])
                    },
                    'conflict_location': {
                        'x': float(pos2[0]),
                        'y': float(pos2[1]),
                        'z': float(pos2[2])
                    }
                }
                
                conflicts.append(conflict)
    
    return conflicts

# ============================================================================
# 3. SPATIAL INDEXING (Octree for fast collision detection)
# ============================================================================

class Octree:
    """Octree for spatial indexing of trajectory segments"""
    
    def __init__(self, center, size, max_depth=6, max_objects=10):
        self.center = np.array(center, dtype=np.float32)
        self.size = size
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.objects = []
        self.children = None
        self.depth = 0
    
    def insert(self, segment):
        """Insert segment into octree"""
        if self.children is not None:
            # Insert into children
            octant = self._get_octant(segment)
            if octant is not None:
                self.children[octant].insert(segment)
                return
        
        self.objects.append(segment)
        
        # Subdivide if necessary
        if len(self.objects) > self.max_objects and self.depth < self.max_depth:
            self._subdivide()
    
    def _get_octant(self, segment):
        """Get octant index for segment"""
        aabb_min, aabb_max = segment.aabb
        center_pos = (aabb_min[:3] + aabb_max[:3]) * 0.5
        
        octant = 0
        if center_pos[0] > self.center[0]: octant |= 1
        if center_pos[1] > self.center[1]: octant |= 2
        if center_pos[2] > self.center[2]: octant |= 4
        
        return octant
    
    def _subdivide(self):
        """Subdivide octree node"""
        if self.children is not None:
            return
        
        half_size = self.size * 0.5
        self.children = []
        
        for i in range(8):
            offset = np.array([
                half_size if i & 1 else -half_size,
                half_size if i & 2 else -half_size,
                half_size if i & 4 else -half_size
            ], dtype=np.float32)
            
            child = Octree(
                self.center + offset,
                half_size,
                self.max_depth,
                self.max_objects
            )
            child.depth = self.depth + 1
            self.children.append(child)
        
        # Redistribute objects
        for obj in self.objects:
            octant = self._get_octant(obj)
            if octant is not None:
                self.children[octant].insert(obj)
        
        self.objects = []
    
    def query(self, aabb):
        """Query segments intersecting with AABB"""
        results = []
        
        # Check if AABB intersects with this node
        aabb_min, aabb_max = aabb
        node_min = self.center - self.size
        node_max = self.center + self.size
        
        if not (np.all(aabb_min[:3] <= node_max) and np.all(node_min <= aabb_max[:3])):
            return results
        
        # Add objects in this node
        results.extend(self.objects)
        
        # Query children
        if self.children is not None:
            for child in self.children:
                results.extend(child.query(aabb))
        
        return results

# ============================================================================
# 4. PARALLEL CONFLICT DETECTOR
# ============================================================================

class ParallelConflictDetector:
    """Highly optimized parallel conflict detector"""
    
    def __init__(self, safety_radius=10.0, num_workers=None):
        self.safety_radius = safety_radius
        self.num_workers = num_workers or mp.cpu_count()
        self.drones = {}
        self.segments = []
        self.conflicts = []
        self.octree = None
        
        self.performance_stats = {
            'load_time': 0,
            'indexing_time': 0,
            'detection_time': 0,
            'num_segments': 0,
            'num_drones': 0,
            'num_workers': self.num_workers
        }
        
        print(f"🚀 Parallel Detector initialized with {self.num_workers} workers")
    
    def load_from_json(self, primary_file: str, existing_file: str, max_drones=1000):
        """Load missions from JSON files with parallel processing"""
        start_time = time.time()
        
        print(f"📂 Loading missions (max: {max_drones})...")
        
        # Load primary mission
        with open(primary_file, 'r') as f:
            primary_data = json.load(f)
        
        primary_drone = self._parse_drone_data(primary_data, is_primary=True)
        self.drones[primary_drone['id']] = primary_drone
        
        # Load existing missions
        try:
            with open(existing_file, 'r') as f:
                existing_data = json.load(f)
            
            if 'drones' in existing_data:
                drones_list = existing_data['drones'][:max_drones]
                print(f"📊 Found {len(drones_list)} drones, processing in parallel...")
                
                # Parallel parsing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for i, drone_data in enumerate(drones_list):
                        future = executor.submit(self._parse_drone_data, drone_data, False)
                        futures.append((i, future))
                    
                    for i, future in futures:
                        drone = future.result()
                        self.drones[drone['id']] = drone
                        if (i + 1) % 100 == 0:
                            print(f"  ✓ Loaded {i+1}/{len(drones_list)} drones")
            else:
                print("⚠️  No 'drones' key found, creating samples...")
                self._create_sample_missions(20)
                
        except Exception as e:
            print(f"❌ Error loading: {e}")
            self._create_sample_missions(20)
        
        self.performance_stats['load_time'] = time.time() - start_time
        self.performance_stats['num_drones'] = len(self.drones)
        self.performance_stats['num_segments'] = len(self.segments)
        
        print(f"✅ Loaded {len(self.drones)} drones in {self.performance_stats['load_time']:.2f}s")
        
        # Build spatial index
        self._build_spatial_index()
    
    def _parse_drone_data(self, drone_data, is_primary=False):
        """Parse drone data from JSON"""
        drone_id = drone_data.get('id', f"drone_{len(self.drones)}")
        
        waypoints = []
        if 'waypoints' in drone_data:
            raw_waypoints = drone_data['waypoints']
            
            for i, wp in enumerate(raw_waypoints):
                t = wp.get('time', wp.get('t', i * 0.1))
                x = wp.get('x', 0)
                y = wp.get('y', 0)
                z = wp.get('z', wp.get('altitude', 50))
                
                vx = vy = vz = 0.0
                if i < len(raw_waypoints) - 1:
                    next_wp = raw_waypoints[i + 1]
                    dt = next_wp.get('t', t + 1) - t
                    if dt > 0:
                        vx = (next_wp.get('x', 0) - x) / dt
                        vy = (next_wp.get('y', 0) - y) / dt
                        vz = (next_wp.get('z', 0) - z) / dt
                
                waypoints.append(Waypoint4D(
                    t=t, x=x, y=y, z=z,
                    vx=vx, vy=vy, vz=vz
                ))
        
        segments = []
        base_id = len(self.segments)
        for i in range(len(waypoints) - 1):
            segment = TrajectorySegment(
                id=base_id + i,
                drone_id=drone_id,
                segment_idx=i,
                start=waypoints[i],
                end=waypoints[i + 1],
                safety_radius=self.safety_radius
            )
            segments.append(segment)
        
        self.segments.extend(segments)
        
        return {
            'id': drone_id,
            'is_primary': is_primary,
            'waypoints': waypoints,
            'segments': segments,
            'start_time': waypoints[0].t if waypoints else 0,
            'end_time': waypoints[-1].t if waypoints else 0
        }
    
    def _create_sample_missions(self, n_drones=20):
        """Create sample missions"""
        for i in range(n_drones):
            drone_id = f"sample_drone_{i}"
            start_time = i * 5
            
            waypoints = []
            for j in range(5):
                t = start_time + j * 10
                x = np.random.uniform(-500, 500)
                y = np.random.uniform(-500, 500)
                z = np.random.uniform(50, 200)
                
                waypoints.append(Waypoint4D(t=t, x=x, y=y, z=z))
            
            segments = []
            base_id = len(self.segments)
            for j in range(len(waypoints) - 1):
                segment = TrajectorySegment(
                    id=base_id + j,
                    drone_id=drone_id,
                    segment_idx=j,
                    start=waypoints[j],
                    end=waypoints[j + 1],
                    safety_radius=self.safety_radius
                )
                segments.append(segment)
            
            self.segments.extend(segments)
            
            self.drones[drone_id] = {
                'id': drone_id,
                'is_primary': False,
                'waypoints': waypoints,
                'segments': segments,
                'start_time': waypoints[0].t,
                'end_time': waypoints[-1].t
            }
    
    def _build_spatial_index(self):
        """Build octree spatial index"""
        start_time = time.time()
        print("🌳 Building spatial index (Octree)...")
        
        if not self.segments:
            return
        
        # Calculate world bounds
        all_points = []
        for seg in self.segments:
            all_points.extend([
                [seg.start.x, seg.start.y, seg.start.z],
                [seg.end.x, seg.end.y, seg.end.z]
            ])
        
        all_points = np.array(all_points)
        center = np.mean(all_points, axis=0)
        max_dist = np.max(np.abs(all_points - center))
        
        # Build octree
        self.octree = Octree(center, max_dist * 1.5, max_depth=8, max_objects=20)
        
        for seg in self.segments:
            self.octree.insert(seg)
        
        self.performance_stats['indexing_time'] = time.time() - start_time
        print(f"✅ Spatial index built in {self.performance_stats['indexing_time']:.2f}s")
    
    def detect_conflicts_parallel(self):
        """Detect conflicts using parallel processing"""
        start_time = time.time()
        
        print(f"🔍 Detecting conflicts (parallel, {self.num_workers} workers)...")
        
        # Find primary drone
        primary_drone = None
        for drone in self.drones.values():
            if drone.get('is_primary', False):
                primary_drone = drone
                break
        
        if not primary_drone:
            print("❌ No primary drone found!")
            return
        
        primary_segments = primary_drone['segments']
        
        # Collect all other drone segments
        other_drones_segments = {}
        for drone_id, drone in self.drones.items():
            if not drone.get('is_primary', False):
                other_drones_segments[drone_id] = drone['segments']
        
        print(f"📊 Checking {len(primary_segments)} primary segments against {len(other_drones_segments)} drones")
        
        # Parallel conflict detection
        conflicts_list = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Submit batch jobs
            for drone_id, other_segments in other_drones_segments.items():
                if not other_segments:
                    continue
                
                future = executor.submit(
                    batch_conflict_detection,
                    primary_segments,
                    other_segments,
                    self.safety_radius
                )
                futures.append(future)
            
            # Collect results with progress
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                conflicts = future.result()
                conflicts_list.extend(conflicts)
                completed += 1
                
                if completed % 10 == 0 or completed == total:
                    print(f"  ⏳ Progress: {completed}/{total} drones checked, {len(conflicts_list)} conflicts found")
        
        self.conflicts = conflicts_list
        
        self.performance_stats['detection_time'] = time.time() - start_time
        
        print(f"✅ Detection complete in {self.performance_stats['detection_time']:.2f}s")
        print(f"🎯 Found {len(self.conflicts)} total conflicts")
    
    def generate_visualizations(self):
        """Generate all visualizations in parallel"""
        print("\n" + "="*80)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create PDF report
        pdf_file = 'drone_conflict_analysis_report.pdf'
        
        with PdfPages(pdf_file) as pdf:
            # Generate all figures in parallel using threads
            print("📊 Creating visualizations in parallel...")
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'summary': executor.submit(self._create_summary_page),
                    '2d': executor.submit(VisualizationEngine.create_2d_trajectory_map, self.drones, self.conflicts),
                    '3d': executor.submit(VisualizationEngine.create_3d_trajectory_plot, self.drones, self.conflicts),
                    '4d': executor.submit(VisualizationEngine.create_4d_static_plot, self.drones, self.conflicts),
                    'heatmap': executor.submit(VisualizationEngine.create_conflict_heatmap, self.drones, self.conflicts),
                }
                
                # Collect and save in order
                print("  ✓ Creating summary page...")
                fig_summary = futures['summary'].result()
                pdf.savefig(fig_summary, bbox_inches='tight', dpi=150)
                plt.close('all')
                del fig_summary
                
                print("  ✓ Creating 2D trajectory maps...")
                fig_2d = futures['2d'].result()
                pdf.savefig(fig_2d, bbox_inches='tight', dpi=150)
                plt.close('all')
                del fig_2d
                
                print("  ✓ Creating 3D trajectory plot...")
                fig_3d = futures['3d'].result()
                pdf.savefig(fig_3d, bbox_inches='tight', dpi=150)
                plt.close('all')
                del fig_3d
                
                print("  ✓ Creating 4D analysis...")
                fig_4d = futures['4d'].result()
                pdf.savefig(fig_4d, bbox_inches='tight', dpi=150)
                plt.close('all')
                del fig_4d
                
                print("  ✓ Creating conflict heatmaps...")
                fig_heatmap = futures['heatmap'].result()
                pdf.savefig(fig_heatmap, bbox_inches='tight', dpi=150)
                plt.close('all')
                del fig_heatmap
        
        print(f"\n✅ PDF report saved: {pdf_file}")
        
        # Generate detailed JSON
        self._generate_detailed_json()
        
        # Clean up memory
        gc.collect()
    
    def _create_summary_page(self):
        """Create comprehensive summary page"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('DRONE CONFLICT ANALYSIS SUMMARY', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Calculate statistics
        severities = [c['severity'] for c in self.conflicts] if self.conflicts else []
        critical_count = severities.count('CRITICAL')
        high_count = severities.count('HIGH')
        medium_count = severities.count('MEDIUM')
        low_count = severities.count('LOW')
        
        # Summary text
        summary_text = [
            "=" * 80,
            "COMPREHENSIVE ANALYSIS SUMMARY",
            "=" * 80,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Type: Multi-threaded Parallel Processing",
            "",
            "MISSION STATISTICS:",
            f"  • Total Drones Analyzed: {self.performance_stats['num_drones']:,}",
            f"  • Trajectory Segments: {self.performance_stats['num_segments']:,}",
            f"  • Safety Radius: {self.safety_radius} meters",
            f"  • Parallel Workers: {self.performance_stats['num_workers']}",
            "",
            "CONFLICT DETECTION RESULTS:",
            f"  • Status: {'⚠️  CONFLICTS DETECTED' if self.conflicts else '✅ CLEAR'}",
            f"  • Total Conflicts Found: {len(self.conflicts):,}",
        ]
        
        if self.conflicts:
            summary_text.extend([
                "",
                "SEVERITY BREAKDOWN:",
                f"  🔴 Critical Conflicts: {critical_count:,}",
                f"  🟠 High Severity: {high_count:,}",
                f"  🟡 Medium Severity: {medium_count:,}",
                f"  🟢 Low Severity: {low_count:,}",
                "",
                "CONFLICT TIMING:",
            ])
            
            times = [c['time'] for c in self.conflicts]
            distances = [c['distance'] for c in self.conflicts]
            
            summary_text.extend([
                f"  • Earliest Conflict: {min(times):.1f}s",
                f"  • Latest Conflict: {max(times):.1f}s",
                f"  • Average Time: {np.mean(times):.1f}s",
                "",
                "SEPARATION DISTANCES:",
                f"  • Minimum Distance: {min(distances):.2f}m",
                f"  • Maximum Distance: {max(distances):.2f}m",
                f"  • Average Distance: {np.mean(distances):.2f}m",
            ])
        
        total_time = sum(self.performance_stats.values())
        
        summary_text.extend([
            "",
            "PERFORMANCE METRICS:",
            f"  • Total Analysis Time: {total_time:.2f}s",
            f"  • Data Loading: {self.performance_stats['load_time']:.2f}s",
            f"  • Spatial Indexing: {self.performance_stats['indexing_time']:.2f}s",
            f"  • Conflict Detection: {self.performance_stats['detection_time']:.2f}s",
            f"  • Segments per Second: {self.performance_stats['num_segments'] / max(total_time, 0.001):,.0f}",
            "",
            "GENERATED OUTPUTS:",
            "  1. drone_conflict_analysis_report.pdf (This document)",
            "  2. detailed_analysis.json (Complete conflict data)",
            "  3. Interactive 4D OpenGL visualization",
            "",
            "ALGORITHMS & OPTIMIZATIONS:",
            "  • 4D Continuous Closest Point of Approach (CPA)",
            "  • Octree spatial indexing for O(log n) queries",
            "  • Multi-process parallel conflict detection",
            "  • AABB culling for early rejection",
            f"  • JIT compilation: {'✓ Enabled (Numba)' if NUMBA_AVAILABLE else '✗ Disabled'}",
            "",
        ])
        
        # Risk assessment
        if critical_count > 0 or high_count > 0:
            summary_text.extend([
                "🚨 RISK ASSESSMENT: HIGH RISK",
                f"   {critical_count + high_count} critical/high severity conflicts detected",
                "   ⚠️  MISSION REPLANNING REQUIRED",
                "   Recommend immediate trajectory adjustment",
            ])
        elif medium_count > 0:
            summary_text.extend([
                "⚠️  RISK ASSESSMENT: MEDIUM RISK",
                f"   {medium_count} medium severity conflicts detected",
                "   Review and consider replanning",
            ])
        elif low_count > 0:
            summary_text.extend([
                "✅ RISK ASSESSMENT: LOW RISK",
                f"   {low_count} low severity conflicts detected",
                "   Mission acceptable with monitoring",
            ])
        else:
            summary_text.extend([
                "✅ RISK ASSESSMENT: NO RISK",
                "   No conflicts detected",
                "   Mission CLEARED for execution",
            ])
        
        summary_text.extend([
            "",
            "=" * 80,
        ])
        
        # Create text box
        plt.figtext(0.05, 0.90, '\n'.join(summary_text), fontsize=8, 
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.axis('off')
        return fig
    
    def _generate_detailed_json(self):
        """Generate comprehensive JSON output"""
        print("📝 Generating detailed JSON analysis...")
        
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'safety_radius': self.safety_radius,
                'total_drones': len(self.drones),
                'total_segments': len(self.segments),
                'total_conflicts': len(self.conflicts),
                'parallel_workers': self.performance_stats['num_workers'],
                'numba_enabled': NUMBA_AVAILABLE,
                'opengl_available': OPENGL_AVAILABLE,
            },
            'performance': self.performance_stats,
            'conflicts': self.conflicts,
            'statistics': {}
        }
        
        # Conflict statistics
        if self.conflicts:
            distances = [c['distance'] for c in self.conflicts]
            times = [c['time'] for c in self.conflicts]
            severities = [c['severity'] for c in self.conflicts]
            
            output['statistics'] = {
                'distance_stats': {
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances)),
                    'mean': float(np.mean(distances)),
                    'median': float(np.median(distances)),
                    'std': float(np.std(distances)),
                },
                'time_stats': {
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'mean': float(np.mean(times)),
                    'median': float(np.median(times)),
                },
                'severity_distribution': {
                    'CRITICAL': severities.count('CRITICAL'),
                    'HIGH': severities.count('HIGH'),
                    'MEDIUM': severities.count('MEDIUM'),
                    'LOW': severities.count('LOW'),
                },
            }
        
        with open('detailed_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("✅ Detailed analysis saved: detailed_analysis.json")

# ============================================================================
# 5. VISUALIZATION ENGINE (Same as before but optimized)
# ============================================================================

class VisualizationEngine:
    """Visualization engine - optimized for parallel generation"""
    
    @staticmethod
    def create_2d_trajectory_map(drones, conflicts=None, figsize=(14, 11)):
        """Create comprehensive 2D trajectory map"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('2D TRAJECTORY ANALYSIS', fontsize=16, fontweight='bold')
        
        ax1, ax2, ax3, ax4 = axes.flat
        
        ax1.set_title('Top View (XY Plane)', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Side View (XZ Plane)', fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Altitude (m)')
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Front View (YZ Plane)', fontweight='bold')
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Altitude (m)')
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Time-Altitude Profile', fontweight='bold')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Altitude (m)')
        ax4.grid(True, alpha=0.3)
        
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(drones))))
        
        for (drone_id, drone_data), color in zip(drones.items(), colors):
            waypoints = drone_data['waypoints']
            if len(waypoints) < 2:
                continue
            
            xs = np.array([wp.x for wp in waypoints])
            ys = np.array([wp.y for wp in waypoints])
            zs = np.array([wp.z for wp in waypoints])
            ts = np.array([wp.t for wp in waypoints])
            
            is_primary = drone_data.get('is_primary', False)
            lw = 3 if is_primary else 1
            ls = '-' if is_primary else '--'
            alpha = 1.0 if is_primary else 0.5
            
            label = f"PRIMARY: {drone_id}" if is_primary else None
            
            ax1.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, alpha=alpha, label=label)
            ax2.plot(xs, zs, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax3.plot(ys, zs, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax4.plot(ts, zs, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            
            if is_primary:
                for ax, x_data, y_data in [(ax1, xs, ys), (ax2, xs, zs), 
                                            (ax3, ys, zs), (ax4, ts, zs)]:
                    ax.scatter(x_data[0], y_data[0], color='green', s=100, marker='o', 
                              edgecolors='black', linewidth=2, zorder=10)
                    ax.scatter(x_data[-1], y_data[-1], color='red', s=100, marker='s', 
                              edgecolors='black', linewidth=2, zorder=10)
        
        # Plot conflicts
        if conflicts:
            severity_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 
                             'MEDIUM': 'yellow', 'LOW': 'green'}
            severity_sizes = {'CRITICAL': 200, 'HIGH': 150, 'MEDIUM': 100, 'LOW': 50}
            
            for conflict in conflicts:
                sev = conflict['severity']
                loc = conflict['location']
                color = severity_colors.get(sev, 'gray')
                size = severity_sizes.get(sev, 50)
                
                ax1.scatter(loc['x'], loc['y'], color=color, s=size, marker='X',
                          edgecolors='black', linewidth=2, zorder=15, alpha=0.8)
                ax2.scatter(loc['x'], loc['z'], color=color, s=size, marker='X',
                          edgecolors='black', linewidth=2, zorder=15, alpha=0.8)
                ax3.scatter(loc['y'], loc['z'], color=color, s=size, marker='X',
                          edgecolors='black', linewidth=2, zorder=15, alpha=0.8)
                ax4.scatter(conflict['time'], loc['z'], color=color, s=size, marker='X',
                          edgecolors='black', linewidth=2, zorder=15, alpha=0.8)
        
        ax1.legend(loc='best', fontsize=8)
        
        # Add conflict legend
        if conflicts:
            legend_elements = [
                Line2D([0], [0], marker='X', color='w', label='CRITICAL',
                      markerfacecolor='red', markersize=10, markeredgecolor='black'),
                Line2D([0], [0], marker='X', color='w', label='HIGH',
                      markerfacecolor='orange', markersize=8, markeredgecolor='black'),
                Line2D([0], [0], marker='X', color='w', label='MEDIUM',
                      markerfacecolor='yellow', markersize=6, markeredgecolor='black'),
                Line2D([0], [0], marker='X', color='w', label='LOW',
                      markerfacecolor='green', markersize=4, markeredgecolor='black')
            ]
            ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_3d_trajectory_plot(drones, conflicts=None, figsize=(15, 10)):
        """Create 3D trajectory plot"""
        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D TRAJECTORY VISUALIZATION', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontweight='bold')
        ax.set_ylabel('Y (m)', fontweight='bold')
        ax.set_zlabel('Altitude (m)', fontweight='bold')
        
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(drones))))
        
        for (drone_id, drone_data), color in zip(drones.items(), colors):
            waypoints = drone_data['waypoints']
            if len(waypoints) < 2:
                continue
            
            xs = [wp.x for wp in waypoints]
            ys = [wp.y for wp in waypoints]
            zs = [wp.z for wp in waypoints]
            
            is_primary = drone_data.get('is_primary', False)
            lw = 3 if is_primary else 1
            alpha = 1.0 if is_primary else 0.5
            
            label = f"PRIMARY: {drone_id}" if is_primary else None
            
            ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha, label=label)
            
            if is_primary:
                ax.scatter([xs[0]], [ys[0]], [zs[0]], color='green', s=150, 
                          marker='o', edgecolors='black', linewidth=2)
                ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', s=150, 
                          marker='s', edgecolors='black', linewidth=2)
        
        # Plot conflicts
        if conflicts:
            severity_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 
                             'MEDIUM': 'yellow', 'LOW': 'green'}
            severity_sizes = {'CRITICAL': 300, 'HIGH': 200, 'MEDIUM': 100, 'LOW': 50}
            
            for conflict in conflicts:
                sev = conflict['severity']
                loc = conflict['location']
                color = severity_colors.get(sev, 'gray')
                size = severity_sizes.get(sev, 50)
                
                ax.scatter([loc['x']], [loc['y']], [loc['z']], 
                          color=color, s=size, marker='X',
                          edgecolors='black', linewidth=2, alpha=0.9)
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        max_range = np.array([
            max([wp.x for d in drones.values() for wp in d['waypoints']]) - 
            min([wp.x for d in drones.values() for wp in d['waypoints']]),
            max([wp.y for d in drones.values() for wp in d['waypoints']]) - 
            min([wp.y for d in drones.values() for wp in d['waypoints']]),
            max([wp.z for d in drones.values() for wp in d['waypoints']]) - 
            min([wp.z for d in drones.values() for wp in d['waypoints']])
        ]).max() / 2.0
        
        mid_x = (max([wp.x for d in drones.values() for wp in d['waypoints']]) + 
                 min([wp.x for d in drones.values() for wp in d['waypoints']])) * 0.5
        mid_y = (max([wp.y for d in drones.values() for wp in d['waypoints']]) + 
                 min([wp.y for d in drones.values() for wp in d['waypoints']])) * 0.5
        mid_z = (max([wp.z for d in drones.values() for wp in d['waypoints']]) + 
                 min([wp.z for d in drones.values() for wp in d['waypoints']])) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_4d_static_plot(drones, conflicts=None, figsize=(16, 12)):
        """Create static 4D visualization"""
        fig = plt.figure(figsize=figsize)
        fig.suptitle('4D SPATIO-TEMPORAL ANALYSIS', fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 3D trajectory with time coloring
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax1.set_title('3D Trajectories (Time Gradient)', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        
        # Time-distance plot
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Separation vs Time', fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (m)')
        ax2.grid(True, alpha=0.3)
        
        # Velocity plot
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_title('Velocity Profile', fontweight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.grid(True, alpha=0.3)
        
        # 4D projection
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_title('4D Projection (XY with Time & Altitude)', fontweight='bold')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.grid(True, alpha=0.3)
        
        for drone_id, drone_data in drones.items():
            waypoints = drone_data['waypoints']
            if len(waypoints) < 2:
                continue
            
            xs = np.array([wp.x for wp in waypoints])
            ys = np.array([wp.y for wp in waypoints])
            zs = np.array([wp.z for wp in waypoints])
            ts = np.array([wp.t for wp in waypoints])
            
            is_primary = drone_data.get('is_primary', False)
            
            # Plot 3D with time coloring - use scatter plot colored by time
            from matplotlib.cm import ScalarMappable
            from matplotlib import cm
            
            # Normalize time for coloring
            if len(ts) > 1 and (ts.max() - ts.min()) > 0:
                t_norm = (ts - ts.min()) / (ts.max() - ts.min())
            else:
                t_norm = np.ones_like(ts) * 0.5
            
            # Create color map
            colors = cm.viridis(t_norm)
            
            # Plot trajectory as line
            ax1.plot(xs, ys, zs, linewidth=3 if is_primary else 1, 
                    alpha=0.8 if is_primary else 0.4, color='red' if is_primary else 'blue')
            
            # Plot points colored by time
            scatter_3d = ax1.scatter(xs, ys, zs, c=ts, cmap='viridis', s=50 if is_primary else 20,
                                    alpha=0.8, edgecolors='black' if is_primary else 'none',
                                    linewidth=1)
            
            # Plot time-distance
            if len(waypoints) > 1:
                distances = []
                for i in range(1, len(waypoints)):
                    dx = waypoints[i].x - waypoints[i-1].x
                    dy = waypoints[i].y - waypoints[i-1].y
                    dz = waypoints[i].z - waypoints[i-1].z
                    distances.append(np.sqrt(dx*dx + dy*dy + dz*dz))
                
                ax2.plot(ts[1:], distances, label=drone_id[:10] if is_primary else None, 
                        linewidth=2 if is_primary else 1,
                        color='red' if is_primary else None)
            
            # Plot velocity
            velocities = []
            for wp in waypoints:
                v = np.sqrt(wp.vx**2 + wp.vy**2 + wp.vz**2)
                velocities.append(v)
            
            ax3.plot(ts, velocities, label=drone_id[:10] if is_primary else None,
                    linewidth=2 if is_primary else 1,
                    color='red' if is_primary else None)
            
            # Plot 4D projection
            scatter = ax4.scatter(xs, ys, c=ts, s=zs/2, cmap='plasma', 
                                 alpha=0.8 if is_primary else 0.4,
                                 edgecolors='black' if is_primary else 'none',
                                 linewidth=1)
        
        # Plot conflicts
        if conflicts:
            conflict_times = [c['time'] for c in conflicts]
            conflict_locs = [[c['location']['x'], c['location']['y'], c['location']['z']] 
                           for c in conflicts]
            severities = [c['severity'] for c in conflicts]
            
            severity_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 
                             'MEDIUM': 'yellow', 'LOW': 'green'}
            colors = [severity_colors.get(s, 'gray') for s in severities]
            
            # Plot in 3D
            for loc, color in zip(conflict_locs, colors):
                ax1.scatter([loc[0]], [loc[1]], [loc[2]], 
                          color=color, s=200, marker='X',
                          edgecolors='black', linewidth=2)
            
            # Mark conflicts in time plots
            for t in conflict_times:
                ax2.axvline(x=t, color='red', alpha=0.2, linestyle='--')
                ax3.axvline(x=t, color='red', alpha=0.2, linestyle='--')
        
        # Add colorbars
        if len(drones) > 0:
            try:
                # Try to add colorbar if scatter plot exists
                from matplotlib import cm
                cbar1 = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax1, label='Time (s)', shrink=0.8)
                cbar2 = plt.colorbar(scatter, ax=ax4, label='Time (s)', shrink=0.8)
            except:
                pass  # Skip colorbar if it fails
        
        # Add legends
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=8, loc='upper right')
        if ax3.get_legend_handles_labels()[0]:
            ax3.legend(fontsize=8, loc='upper right')
        
        # Add conflict legend if conflicts exist
        if conflicts:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='CRITICAL',
                      markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='HIGH',
                      markerfacecolor='orange', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='MEDIUM',
                      markerfacecolor='yellow', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='LOW',
                      markerfacecolor='green', markersize=4)
            ]
            ax4.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_conflict_heatmap(drones, conflicts, figsize=(14, 10)):
        """Create conflict density heatmaps"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('CONFLICT DENSITY ANALYSIS', fontsize=16, fontweight='bold')
        
        if not conflicts:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No Conflicts Detected', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                ax.set_axis_off()
            return fig
        
        # Extract conflict data
        xs = np.array([c['location']['x'] for c in conflicts])
        ys = np.array([c['location']['y'] for c in conflicts])
        zs = np.array([c['location']['z'] for c in conflicts])
        ts = np.array([c['time'] for c in conflicts])
        severities = [c['severity'] for c in conflicts]
        
        # XY heatmap
        ax1 = axes[0, 0]
        h1 = ax1.hist2d(xs, ys, bins=25, cmap='YlOrRd', cmin=1)
        plt.colorbar(h1[3], ax=ax1, label='Conflict Count')
        ax1.set_title('XY Plane Conflict Density', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.2)
        
        # XZ heatmap
        ax2 = axes[0, 1]
        h2 = ax2.hist2d(xs, zs, bins=25, cmap='YlOrRd', cmin=1)
        plt.colorbar(h2[3], ax=ax2, label='Conflict Count')
        ax2.set_title('XZ Plane Conflict Density', fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Altitude (m)')
        ax2.grid(True, alpha=0.2)
        
        # Time-Altitude heatmap
        ax3 = axes[1, 0]
        h3 = ax3.hist2d(ts, zs, bins=25, cmap='viridis', cmin=1)
        plt.colorbar(h3[3], ax=ax3, label='Conflict Count')
        ax3.set_title('Time-Altitude Conflict Density', fontweight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.grid(True, alpha=0.2)
        
        # Severity distribution
        ax4 = axes[1, 1]
        severity_counts = {s: severities.count(s) for s in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']}
        colors_map = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green'}
        
        bars = ax4.bar(severity_counts.keys(), severity_counts.values(),
                      color=[colors_map[k] for k in severity_counts.keys()],
                      edgecolor='black', linewidth=2)
        
        ax4.set_title('Conflict Severity Distribution', fontweight='bold')
        ax4.set_xlabel('Severity Level')
        ax4.set_ylabel('Count')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig

# ============================================================================
# 6. ENHANCED 4D OPENGL VISUALIZER
# ============================================================================

if OPENGL_AVAILABLE:
    class Enhanced4DVisualizer:
        """Enhanced OpenGL 4D visualization with better controls"""
        
        def __init__(self, drones, conflicts=None):
            self.drones = drones
            self.conflicts = conflicts or []
            self.display = (1400, 900)
            self.rotation = [30, 45, 0]
            self.translation = [0, 0, -100]
            self.current_time = 0.0
            self.max_time = self._get_max_time()
            self.animation_speed = 1.0
            self.paused = True
            self.show_trails = True
            self.show_conflicts = True
            
            pygame.init()
            pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL)
            pygame.display.set_caption("4D Drone Conflict Visualization - Enhanced")
            
            gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 1000.0)
            glTranslatef(*self.translation)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16, bold=True)
            
            print("\n" + "="*80)
            print("4D VISUALIZATION CONTROLS")
            print("="*80)
            print("SPACE      - Play/Pause animation")
            print("UP/DOWN    - Increase/decrease speed")
            print("R          - Reset time to start")
            print("T          - Toggle trajectory trails")
            print("C          - Toggle conflict markers")
            print("W/S        - Zoom in/out")
            print("Mouse Drag - Rotate view")
            print("ESC        - Exit")
            print("="*80 + "\n")
            
            self.run()
        
        def _get_max_time(self):
            max_t = 0
            for drone_data in self.drones.values():
                waypoints = drone_data['waypoints']
                if waypoints:
                    max_t = max(max_t, max(wp.t for wp in waypoints))
            return max_t
        
        def draw_text(self, text, x, y, color=(255, 255, 255)):
            """Draw text overlay"""
            text_surface = self.font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            
            glWindowPos2d(x, y)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        def draw_grid(self, size=500, divisions=10):
            """Draw reference grid"""
            glColor4f(0.3, 0.3, 0.3, 0.3)
            glBegin(GL_LINES)
            
            step = size / divisions
            for i in range(-divisions, divisions + 1):
                pos = i * step
                # X lines
                glVertex3f(pos, -size, 0)
                glVertex3f(pos, size, 0)
                # Y lines
                glVertex3f(-size, pos, 0)
                glVertex3f(size, pos, 0)
            
            glEnd()
        
        def draw_axes(self, length=50):
            """Draw coordinate axes"""
            glLineWidth(3.0)
            glBegin(GL_LINES)
            
            # X axis - Red
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(length, 0, 0)
            
            # Y axis - Green
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, length, 0)
            
            # Z axis - Blue
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, length)
            
            glEnd()
            glLineWidth(1.0)
        
        def draw_trajectory(self, waypoints, is_primary=False):
            """Draw trajectory with time gradient"""
            if len(waypoints) < 2:
                return
            
            times = [wp.t for wp in waypoints]
            t_min, t_max = min(times), max(times)
            
            glLineWidth(4.0 if is_primary else 2.0)
            
            if self.show_trails:
                glBegin(GL_LINE_STRIP)
                for wp in waypoints:
                    # Time-based coloring
                    if t_max > t_min:
                        t_norm = (wp.t - t_min) / (t_max - t_min)
                    else:
                        t_norm = 0.5
                    
                    # Color gradient: blue -> cyan -> green -> yellow -> red
                    if t_norm < 0.25:
                        r, g, b = 0, t_norm * 4, 1
                    elif t_norm < 0.5:
                        r, g, b = 0, 1, 1 - (t_norm - 0.25) * 4
                    elif t_norm < 0.75:
                        r, g, b = (t_norm - 0.5) * 4, 1, 0
                    else:
                        r, g, b = 1, 1 - (t_norm - 0.75) * 4, 0
                    
                    alpha = 0.9 if is_primary else 0.5
                    glColor4f(r, g, b, alpha)
                    glVertex3f(wp.x, wp.y, wp.z)
                
                glEnd()
            
            # Draw current position
            for i in range(len(waypoints) - 1):
                wp1, wp2 = waypoints[i], waypoints[i + 1]
                
                if wp1.t <= self.current_time <= wp2.t:
                    alpha = (self.current_time - wp1.t) / max(wp2.t - wp1.t, 0.001)
                    x = wp1.x + alpha * (wp2.x - wp1.x)
                    y = wp1.y + alpha * (wp2.y - wp1.y)
                    z = wp1.z + alpha * (wp2.z - wp1.z)
                    
                    glPushMatrix()
                    glTranslatef(x, y, z)
                    
                    if is_primary:
                        glColor4f(1, 1, 0, 1.0)  # Yellow
                        radius = 3.0
                    else:
                        glColor4f(0.3, 0.7, 1, 0.9)  # Light blue
                        radius = 1.5
                    
                    quad = gluNewQuadric()
                    gluSphere(quad, radius, 16, 16)
                    gluDeleteQuadric(quad)
                    
                    glPopMatrix()
                    break
            
            glLineWidth(1.0)
        
        def draw_conflicts(self):
            """Draw conflict markers"""
            if not self.show_conflicts:
                return
            
            severity_colors = {
                'CRITICAL': (1, 0, 0, 0.9),
                'HIGH': (1, 0.5, 0, 0.8),
                'MEDIUM': (1, 1, 0, 0.7),
                'LOW': (0, 1, 0, 0.6)
            }
            
            severity_sizes = {
                'CRITICAL': 4.0,
                'HIGH': 3.0,
                'MEDIUM': 2.0,
                'LOW': 1.5
            }
            
            for conflict in self.conflicts:
                if abs(conflict['time'] - self.current_time) < 20.0:
                    loc = conflict['location']
                    sev = conflict['severity']
                    
                    color = severity_colors.get(sev, (0.5, 0.5, 0.5, 0.5))
                    size = severity_sizes.get(sev, 1.0)
                    
                    glPushMatrix()
                    glTranslatef(loc['x'], loc['y'], loc['z'])
                    glColor4f(*color)
                    
                    quad = gluNewQuadric()
                    gluSphere(quad, size, 12, 12)
                    gluDeleteQuadric(quad)
                    
                    glPopMatrix()
        
        def handle_events(self):
            """Handle user input"""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.animation_speed *= 1.5
                    elif event.key == pygame.K_DOWN:
                        self.animation_speed /= 1.5
                    elif event.key == pygame.K_r:
                        self.current_time = 0.0
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                    elif event.key == pygame.K_c:
                        self.show_conflicts = not self.show_conflicts
                    elif event.key == pygame.K_w:
                        self.translation[2] += 10
                    elif event.key == pygame.K_s:
                        self.translation[2] -= 10
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        self.translation[2] += 5
                    elif event.button == 5:  # Scroll down
                        self.translation[2] -= 5
            
            # Mouse drag for rotation
            if pygame.mouse.get_pressed()[0]:
                dx, dy = pygame.mouse.get_rel()
                self.rotation[0] += dy * 0.5
                self.rotation[1] += dx * 0.5
            else:
                pygame.mouse.get_rel()
            
            return True
        
        def render(self):
            """Render the scene"""
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Apply transformations
            glTranslatef(*self.translation)
            glRotatef(self.rotation[0], 1, 0, 0)
            glRotatef(self.rotation[1], 0, 1, 0)
            
            # Draw grid and axes
            self.draw_grid()
            self.draw_axes()
            
            # Draw trajectories
            for drone_id, drone_data in self.drones.items():
                waypoints = drone_data['waypoints']
                if waypoints:
                    is_primary = drone_data.get('is_primary', False)
                    self.draw_trajectory(waypoints, is_primary)
            
            # Draw conflicts
            self.draw_conflicts()
            
            # Update time
            if not self.paused:
                self.current_time += 0.016 * self.animation_speed * 10
                if self.current_time > self.max_time:
                    self.current_time = 0.0
            
            # Draw UI
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluOrtho2D(0, self.display[0], 0, self.display[1])
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Info panel
            y_offset = self.display[1] - 30
            self.draw_text(f"Time: {self.current_time:.1f}s / {self.max_time:.1f}s", 10, y_offset)
            y_offset -= 25
            self.draw_text(f"Speed: {self.animation_speed:.1f}x", 10, y_offset)
            y_offset -= 25
            status_color = (0, 255, 0) if not self.paused else (255, 0, 0)
            self.draw_text(f"Status: {'PLAYING' if not self.paused else 'PAUSED'}", 10, y_offset, status_color)
            y_offset -= 25
            self.draw_text(f"Conflicts: {len(self.conflicts)}", 10, y_offset, (255, 100, 100))
            y_offset -= 25
            self.draw_text(f"Trails: {'ON' if self.show_trails else 'OFF'}", 10, y_offset)
            y_offset -= 25
            self.draw_text(f"Markers: {'ON' if self.show_conflicts else 'OFF'}", 10, y_offset)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
            pygame.display.flip()
        
        def run(self):
            """Main render loop"""
            running = True
            while running:
                running = self.handle_events()
                self.render()
                self.clock.tick(60)
            
            pygame.quit()

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with optimized parallel processing"""
    print("\n" + "="*80)
    print("ULTRA-OPTIMIZED DRONE CONFLICT DETECTION SYSTEM")
    print("="*80)
    print("Features:")
    print("  🚀 Heavy Multi-threading & Parallel Processing")
    print("  🎯 Octree Spatial Indexing (O(log n) queries)")
    print(f"  ⚡ JIT Compilation: {'✓ Enabled' if NUMBA_AVAILABLE else '✗ Disabled'}")
    print("  📊 2D/3D/4D Comprehensive Visualizations")
    print("  🎮 Interactive OpenGL 4D Viewer")
    print("  📄 Detailed PDF Report Generation")
    print("="*80 + "\n")
    
    # Configuration
    PRIMARY_FILE = "/home/arka/Trajectra/waypoint_generation/primary_waypoint.json"
    EXISTING_FILE = "/home/arka/Trajectra/waypoint_generation/drone_waypoints.json"
    SAFETY_RADIUS = 10.0
    MAX_DRONES = 1000
    
    print(f"Configuration:")
    print(f"  Safety Radius: {SAFETY_RADIUS}m")
    print(f"  Max Drones: {MAX_DRONES}")
    print(f"  CPU Cores: {mp.cpu_count()}")
    print()
    
    # Initialize detector
    detector = ParallelConflictDetector(safety_radius=SAFETY_RADIUS)
    
    try:
        total_start = time.time()
        
        # Load data
        print("="*80)
        print("STEP 1: LOADING MISSION DATA")
        print("="*80)
        detector.load_from_json(PRIMARY_FILE, EXISTING_FILE, max_drones=MAX_DRONES)
        
        # Detect conflicts
        print("\n" + "="*80)
        print("STEP 2: PARALLEL CONFLICT DETECTION")
        print("="*80)
        detector.detect_conflicts_parallel()
        
        # Generate visualizations
        print("\n" + "="*80)
        print("STEP 3: VISUALIZATION GENERATION")
        print("="*80)
        detector.generate_visualizations()
        
        # Add after your detector.generate_visualizations() call
        from interactive_visualization import launch_interactive_viewer

        # Launch interactive viewer with your data
        print("\n🎮 Launching interactive visualizations...")
        launch_interactive_viewer(detector.drones, detector.conflicts, viewer_type='menu')
        
        total_time = time.time() - total_start
        
        # Summary
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total Time: {total_time:.2f}s")
        print(f"Drones: {detector.performance_stats['num_drones']:,}")
        print(f"Segments: {detector.performance_stats['num_segments']:,}")
        print(f"Conflicts: {len(detector.conflicts):,}")
        print(f"Throughput: {detector.performance_stats['num_segments'] / max(total_time, 0.001):,.0f} segments/sec")
        print()
        
        if detector.conflicts:
            severities = [c['severity'] for c in detector.conflicts]
            print("⚠️  CONFLICTS DETECTED:")
            print(f"  🔴 Critical: {severities.count('CRITICAL'):,}")
            print(f"  🟠 High: {severities.count('HIGH'):,}")
            print(f"  🟡 Medium: {severities.count('MEDIUM'):,}")
            print(f"  🟢 Low: {severities.count('LOW'):,}")
        else:
            print("✅ NO CONFLICTS - MISSION CLEAR")
        
        print("\nGenerated Files:")
        print("  📄 drone_conflict_analysis_report.pdf")
        print("  📋 detailed_analysis.json")
        print("="*80 + "\n")
        
        # Ask if user wants interactive visualization
        if OPENGL_AVAILABLE and len(detector.drones) > 0:
            print("🎮 INTERACTIVE VISUALIZATION AVAILABLE")
            print("="*80)
            print("Would you like to launch interactive 4D viewer?")
            print("  1. Yes - Launch 4D Viewer (Animated)")
            print("  2. No - Skip visualization")
            print()
            
            try:
                choice = input("Enter choice (1-2) or press Enter to skip: ").strip()
                
                if choice == '1':
                    print("\n🚀 Launching 4D Interactive Viewer...")
                    Enhanced4DVisualizer(detector.drones, detector.conflicts)
                else:
                    print("Skipping interactive visualization.")
            except KeyboardInterrupt:
                print("\nSkipping interactive visualization.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Please check file paths:")
        print(f"  Primary: {PRIMARY_FILE}")
        print(f"  Existing: {EXISTING_FILE}")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🔧 Required packages:")
    print("  pip install numpy matplotlib PyOpenGL pygame numba")
    print()
    
    main()