

# 🚁 UAV Airspace Simulation Engine

### 4D Spatio-Temporal Multi-Drone Simulation & Collision Visualization

C++17 • OpenGL • GLUT • JSON-Driven Architecture

---

## 1. System Purpose

This project implements a **real-time 4D (x, y, z, t) UAV airspace simulation engine** designed to:

* Simulate large-scale multi-drone trajectory playback
* Replay time-synchronized waypoint missions
* Visualize spatio-temporal conflicts
* Highlight a mission-critical **Primary Drone**
* Render collision severity zones dynamically
* Provide HUD-level operational telemetry

This is not a planner.
This is a **visual simulation + playback engine** driven by precomputed trajectory and conflict analysis JSON data.

---

## 2. Core Capabilities

### 2.1 4D Trajectory Simulation

Each drone follows:

```
(x(t), y(t), z(t))
```

Where time is continuous and interpolated between waypoints.

Supports:

* Time-wrapped trajectories
* High resolution playback
* Subsampled waypoints for large datasets
* Precise position interpolation

---

### 2.2 Large-Scale Multi-Agent Simulation

Supports up to:

```
MAX_DRONES = 1000
```

Each drone has:

* Independent trajectory
* Physics smoothing
* Trail memory buffer
* LOD-based rendering
* Dynamic color state (collision severity)

---

### 2.3 Primary Drone Emphasis System

The primary drone is:

* 40× visually emphasized
* Pulsing green glow
* Rotating searchlight cone
* Long breadcrumb trail
* Waypoint highlighting
* Dedicated HUD panel
* Velocity vector indicator
* Compass heading overlay

This ensures mission-critical visibility even in dense traffic.

---

### 2.4 Collision Visualization (4D)

Collisions are not detected in this engine.
They are **loaded from analysis JSON**.

Each collision includes:

```json
{
  "primary_drone": "...",
  "conflicting_drone": "...",
  "time": float,
  "distance": float,
  "severity": "LOW | MEDIUM | HIGH",
  "location": {x,y,z}
}
```

Visualization includes:

* Severity-colored volumetric spheres
* Pulsing warning ring
* Drone-to-drone vector line
* Distance annotation
* Time annotation
* HUD alert panel

Severity color scheme:

| Severity | Color  |
| -------- | ------ |
| LOW      | Yellow |
| MEDIUM   | Orange |
| HIGH     | Red    |

---

## 3. Architecture

### 3.1 High-Level Architecture

```
+---------------------------+
| Simulation Controller     |
|---------------------------|
| - Time management         |
| - Update loop             |
| - Drone orchestration     |
+------------+--------------+
             |
             v
+---------------------------+
| Drone Entities            |
|---------------------------|
| - Physics state           |
| - Trajectory interpolation|
| - Trail memory            |
| - Visual properties       |
+------------+--------------+
             |
             v
+---------------------------+
| OpenGL Renderer           |
|---------------------------|
| - LOD system              |
| - Lighting system         |
| - HUD rendering           |
| - Collision rendering     |
+------------+--------------+
             |
             v
+---------------------------+
| JSON Loaders              |
|---------------------------|
| - Trajectory Loader       |
| - Collision Loader        |
+---------------------------+
```

---

## 4. Major Components

---

### 4.1 Vector3D

Custom 3D math utility:

* Dot / Cross
* Lerp
* Normalization
* Distance metrics
* JSON serialization

Used throughout physics and rendering.

---

### 4.2 DronePhysics Engine

Implements:

* Force accumulation
* Drag modeling
* Acceleration clamping
* Velocity clamping
* Horizontal vs vertical speed control
* Arrival steering behavior

Provides realistic smoothing over waypoint playback.

---

### 4.3 Trajectory System

Stores:

```
Waypoint(position, timestamp, arrivalRadius)
```

Capabilities:

* Binary search time lookup
* Continuous interpolation
* Time wrapping
* Predictive future sampling

This is where 4D becomes operational.

---

### 4.4 Rendering Engine (OpenGL)

Implements:

* Perspective projection
* Depth testing
* Blending
* Dynamic lighting
* Multi-light system
* Display list caching
* LOD switching

LOD Levels:

* HIGH (full model + rotors)
* MEDIUM
* LOW
* POINT (far distance)

Primary drone is always rendered last to ensure visibility.

---

### 4.5 HUD System

Real-time overlays include:

* Simulation time
* FPS
* Drone count
* Active collisions
* Upcoming conflicts
* Primary drone telemetry
* Waypoint index
* Velocity vector
* Simulation speed

Rendered in orthographic projection layer.

---

## 5. Time System

Simulation loop:

```
deltaTime = real_time_delta
scaledDelta = deltaTime * timeScale
simulationTime += scaledDelta
```

Time wraps at:

```
200 seconds
```

All drones reset at wrap boundary.

Collision lookup:

```
getCollisionsAtTime(t, tolerance=1.0)
```

---

## 6. Performance Design

### 6.1 LOD Optimization

Distance-based rendering reduces geometry load.

### 6.2 Trail Sampling

Regular drones:

* Trail update every 3 frames

Primary drone:

* Every frame
* Breadcrumb markers
* Ground shadow sampling

### 6.3 Waypoint Subsampling

Large waypoint sets reduced to ~200 per drone for performance.

---

## 7. Input System

### Controls

| Key   | Action              |
| ----- | ------------------- |
| SPACE | Pause               |
| R     | Reset               |
| V     | Reset Camera        |
| G     | Toggle Grid         |
| T     | Toggle Trails       |
| C     | Toggle Collisions   |
| L     | Toggle Trajectories |
| H     | Toggle HUD          |
| B     | Toggle Labels       |
| 1-4   | Time Scale          |
| S     | Print Status        |

Mouse:

* Left: Rotate
* Right: Zoom
* Middle: Pan

---

## 8. Build

### Dependencies

* OpenGL
* GLUT
* GLU
* nlohmann/json
* C++17
* pthread

### Compile

```bash
g++ uav_simulation.cpp -o uav_sim \
-lglut -lGLU -lGL \
-std=c++17 -O3 -pthread
```

---

## 9. Data Format

### Primary Drone JSON

```json
{
  "id": "primary_drone",
  "waypoints": [
    {"time":0, "x":0, "y":0, "z":50}
  ]
}
```

### Multi-Drone JSON

```json
{
  "metadata": {
    "num_drones": 100,
    "total_time": 200,
    "time_step": 0.1
  },
  "drones": [...]
}
```

### Collision JSON

```json
{
  "metadata": {...},
  "conflicts": [...]
}
```

---

## 10. What This System Is

* A deterministic 4D playback engine
* A collision visualization framework
* A multi-agent airspace simulator
* A research-grade spatial debugging tool
* A performance-optimized OpenGL visualizer

---

## 11. What This System Is Not

* Not a trajectory planner
* Not a real-time collision avoidance algorithm
* Not a physics-accurate aerodynamic model
* Not a distributed air traffic manager

It visualizes results from those systems.

---

## 12. Conceptual Model

This engine operates in **4D Euclidean space**:

```
R³ × T
```

Collision =

```
|| P_primary(t) - P_drone_i(t) || < safety_radius
```

Where conflict time and severity are precomputed externally.

---
