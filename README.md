# 🚁 UAV Deconfliction System - Enhanced Visualization Suite

## 📋 Overview

This project provides a **comprehensive UAV (drone) conflict detection and visualization system** that simulates drone trajectories, detects potential collisions, and provides interactive 3D/4D visualization. The system is specifically designed to analyze **primary drone missions** against existing drone traffic and identify potential conflicts with enhanced visual representation.

## ✨ Key Features

### 🎯 Core System
- **Primary Drone Detection**: Identifies and analyzes a "primary drone" (medical/special mission) against regular drone traffic
- **4D Conflict Analysis**: Time-aware collision detection using continuous closest point of approach
- **Parallel Processing**: Multi-threaded/process optimization for handling thousands of drones
- **Octree Spatial Indexing**: O(log n) collision detection for efficient large-scale simulations

### 🎨 Enhanced Visualization
- **Primary Drone Highlighting**: 40x larger size, glowing green pulsing sphere with trail
- **Multi-view 2D/3D/4D**: Interactive OpenGL-based visualizations
- **Real-time Conflict Display**: Visual markers for conflict zones with severity coding
- **Advanced UI**: HUD with statistics, performance metrics, and controls

### 📊 Analysis Capabilities
- **JSON-based Trajectories**: Load real or simulated drone waypoints
- **Comprehensive Reporting**: PDF reports with heatmaps, statistics, and risk assessment
- **Export Functionality**: Save visualizations, data, and analysis results



## 🔧 Installation & Dependencies

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: Multi-core processor
- **GPU**: OpenGL 3.0+ compatible (for visualization)
- **RAM**: 8GB minimum, 16GB recommended for large simulations

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update

# Install C++ build tools
sudo apt install build-essential cmake

# Install OpenGL/GLUT libraries
sudo apt install freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install additional Python system dependencies
sudo apt install python3-dev python3-tk
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy matplotlib PyOpenGL PyOpenGL-accelerate pygame
pip install scipy
pip install nlohmann-json  # For C++ JSON support

# For enhanced performance (optional)
pip install numba
```

### Step 3: Compile C++ Components

```bash
# Compile waypoint generator
g++ main.cpp -o generate_waypoints -O3 -std=c++17 -pthread

# Compile main simulation
g++ main.cpp -o uav_sim -lglut -lGLU -lGL -std=c++17 -O3 -pthread

# Make them executable
chmod +x generate_waypoints uav_sim
```

## 🚀 How to Run - Complete Workflow

### **Step 1: Generate Drone Waypoints**

Generate simulated drone trajectories including a primary drone:

```bash
# Generate with default settings (50 regular drones + primary drone)
./generate_waypoints --drones 50 --primary-drone

# Or with custom settings:
./generate_waypoints --drones 100 --formation grid --output drone_waypoints.json --primary-drone-file primary_waypoint.json
```

**Options:**
- `--drones N`: Number of regular drones (1-1000)
- `--formation TYPE`: random/grid/spiral/v/random_walk
- `--output FILE`: Regular drones output file
- `--primary-drone`: Generate primary drone
- `--primary-drone-file FILE`: Primary drone output file

**Output Files:**
- `drone_waypoints.json`: Regular drone trajectories
- `primary_waypoint.json`: Primary drone trajectory

### **Step 2: Run Conflict Analysis & Visualization**

```bash
# Run the comprehensive Python analysis
python3 trajectory_control.py

# Or run specific modes:
python3 trajectory_control.py --mode full     # Full analysis with visualizations
python3 trajectory_control.py --mode quick    # Quick check only
python3 trajectory_control.py --mode test     # Run examples and benchmarks
```

**Python Analysis Features:**
- Loads both regular and primary drone trajectories
- Performs 4D conflict detection
- Generates comprehensive PDF report
- Creates interactive visualizations
- Saves detailed analysis to JSON

### **Step 3: Launch Interactive OpenGL Simulation**

```bash
# Run the enhanced OpenGL simulation
./uav_sim

# The simulation automatically loads:
# - Primary drone from primary_waypoint.json
# - Regular drones from drone_waypoints.json
# - Conflict analysis from detailed_analysis.json
```

**Simulation Controls:**
- **SPACE**: Pause/Resume
- **R**: Reset simulation
- **V**: Reset camera
- **G**: Toggle grid
- **T**: Toggle trails
- **C**: Toggle conflict zones
- **H**: Toggle HUD
- **1-4**: Speed control (0.5x to 5x)
- **Mouse**: Rotate/Zoom/Pan

## 🎮 Visual Features

### Enhanced Primary Drone
The primary drone is visually enhanced to stand out:
- **40x larger** than regular drones
- **Bright green pulsing glow** with searchlight
- **Long glowing trail** showing exact path
- **Compass** showing direction
- **Velocity vector** arrow
- **Enhanced HUD panel** with detailed status

### Visualization Modes
1. **2D Multi-View**: XY, XZ, YZ, and time-altitude plots
2. **3D Interactive**: Full 3D trajectory visualization
3. **4D Analysis**: Spatio-temporal analysis with time as 4th dimension
4. **Conflict Heatmaps**: Density analysis of conflict zones

### UI Elements
- **HUD Panel**: Real-time statistics and warnings
- **Conflict Alerts**: Color-coded by severity (CRITICAL/HIGH/MEDIUM/LOW)
- **Performance Metrics**: FPS, simulation speed, drone counts
- **File Info**: Loaded trajectory and analysis files

## 📊 Output Files

The system generates several output files:

1. **`drone_waypoints.json`**: Regular drone trajectories
2. **`primary_waypoint.json`**: Primary drone trajectory  
3. **`detailed_analysis.json`**: Complete conflict analysis
4. **`drone_conflict_analysis_report.pdf`**: Comprehensive PDF report
5. **Screenshots**: Automatically saved with timestamp

## 🔍 Understanding the Analysis

### Conflict Severity Levels
- **CRITICAL**: < 3m separation (immediate danger)
- **HIGH**: 3-6m separation (high risk)
- **MEDIUM**: 6-10m separation (medium risk)  
- **LOW**: 10-15m separation (low risk)

### Analysis Metrics
- **Closest Point of Approach (CPA)**: Minimum distance between drones
- **Time to CPA**: When the closest approach occurs
- **Relative Velocity**: Speed and direction difference
- **Risk Score**: Calculated based on distance and severity

## 🛠️ Advanced Usage

### Custom Trajectory Loading
```bash
# Load custom drone trajectories
./uav_sim  # Press 'N' then enter path to custom JSON

# Load custom primary drone  
./uav_sim  # Press 'P' then enter path to primary JSON

# Load custom collision analysis
./uav_sim  # Press 'A' then enter path to analysis JSON
```

### Python API Usage
```python
from trajectory_control import check_mission, check_mission_from_files

# Quick check of missions
result = check_mission_from_files(
    primary_file="primary_waypoint.json",
    existing_file="drone_waypoints.json",
    safety_radius=10.0,
    max_drones=100
)

print(f"Status: {result['status']}")
print(f"Conflicts: {result['conflict_count']}")
```

### Batch Processing
```bash
# Generate, analyze, and visualize in one command
./generate_waypoints --drones 100 --primary-drone && \
python3 trajectory_control.py --mode full && \
./uav_sim
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenGL Errors**:
   ```bash
   # Check OpenGL installation
   glxinfo | grep "OpenGL version"
   
   # If missing, install proper drivers
   sudo apt install mesa-utils
   ```

2. **Missing Dependencies**:
   ```bash
   # Reinstall Python dependencies
   pip install --force-reinstall numpy PyOpenGL pygame
   ```

3. **Compilation Errors**:
   ```bash
   # Ensure all libraries are installed
   sudo apt install libglm-dev libglew-dev
   
   # Recompile with verbose output
   g++ main.cpp -o uav_sim -lglut -lGLU -lGL -std=c++17 -O3 -pthread -v
   ```

4. **File Not Found Errors**:
   - Ensure `generate_waypoints` is run first
   - Check file paths in the code match your system

### Performance Tips
- For >500 drones, use `--drones 500` max
- Close other graphics-intensive applications
- Reduce visualization quality if FPS is low
- Use `--mode quick` for faster analysis

## 📈 Example Workflow

Here's a complete example session:

```bash
# 1. Generate drone trajectories
./generate_waypoints --drones 50 --formation spiral --primary-drone

# 2. Analyze for conflicts
python3 trajectory_control.py --mode full

# 3. Launch interactive visualization
./uav_sim

# 4. In the simulation:
#    - Look for the large green pulsing primary drone
#    - Press 'H' to show HUD
#    - Press 'T' to show trails
#    - Press 'C' to show conflict zones
#    - Use mouse to navigate
```

## 🎯 Expected Output

After running all components, you should see:

1. **Terminal Output**:
   - Generation statistics
   - Conflict detection results
   - Performance metrics

2. **Visualization**:
   - Interactive 3D window with drones
   - Primary drone clearly visible
   - Conflict zones marked in red/orange/yellow

3. **Files Created**:
   - JSON files with trajectories and analysis
   - PDF report with plots and statistics
   - Optional screenshots

## 📚 Technical Details

### Algorithms Used
1. **4D Continuous Closest Point of Approach (CPA)**
2. **Octree Spatial Partitioning** for O(log n) collision detection
3. **Reynolds Flocking** for realistic swarm behavior
4. **Predictive Path Planning** for collision avoidance
5. **Parallel Processing** with multiprocessing pools

### Data Structures
- **Waypoint4D**: 4D waypoints with time
- **TrajectorySegment**: Continuous trajectory segments
- **Octree**: Spatial indexing structure
- **Display List Cache**: OpenGL optimization

### Performance Characteristics
- **Small scale** (<100 drones): Real-time visualization
- **Medium scale** (100-500 drones): Near real-time
- **Large scale** (500-1000 drones): Requires optimization
