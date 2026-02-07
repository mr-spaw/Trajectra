#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <set>
#include <string>
#include <random>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <queue>
#include <chrono>

using json = nlohmann::json;

// Configuration constants
const float TIME_STEP = 0.1f;  // 10 Hz update rate
const float TOTAL_TIME = 200.0f;
const int NUM_WAYPOINTS = static_cast<int>(TOTAL_TIME / TIME_STEP) + 1;
const int DEFAULT_NUM_DRONES = 149; //number of scheduled drones
const int MAX_DRONES = 1000;

// Safe bounds for coordinates (meters)
const float X_MIN = -800.0f;
const float X_MAX = 800.0f;
const float Y_MIN = -800.0f;
const float Y_MAX = 800.0f;
const float Z_MIN = 30.0f;    // Minimum safe altitude
const float Z_MAX = 300.0f;   // Maximum altitude

// Collision avoidance parameters
const float SAFETY_RADIUS = 5.0f;          // Physical drone radius
const float MIN_SEPARATION = 10.0f;        // Minimum separation distance
const float CRITICAL_SEPARATION = 7.0f;    // Critical distance for emergency avoidance
const float PERSONAL_SPACE = 15.0f;        // Preferred personal space
const float VERTICAL_BUFFER = 8.0f;        // Extra vertical separation

// Drone performance characteristics
const float MAX_VELOCITY = 15.0f;          // Maximum cruise speed (m/s)
const float MAX_ACCELERATION = 3.0f;       // Normal acceleration (m/s²)
const float MAX_DECELERATION = 4.0f;       // Emergency deceleration
const float MAX_ASCENT_RATE = 5.0f;        // Maximum climb rate (m/s)
const float MAX_DESCENT_RATE = 4.0f;       // Maximum descent rate (m/s)
const float TURN_RATE = 30.0f * M_PI / 180.0f;  // 30 degrees per second

// Swarm behavior parameters
const float SWARM_COHESION = 0.3f;         // Tendency to stay with group
const float SWARM_SEPARATION = 0.4f;       // Tendency to avoid neighbors
const float SWARM_ALIGNMENT = 0.3f;        // Tendency to align with neighbors
const float SWARM_RADIUS = 50.0f;          // Communication/neighborhood radius

// Formation patterns
enum FormationType {
    FORMATION_RANDOM,
    FORMATION_GRID,
    FORMATION_SPIRAL,
    FORMATION_V,
    FORMATION_RANDOM_WALK
};

struct Vector3D {
    float x, y, z;
    
    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    Vector3D normalized() const {
        float len = length();
        if (len > 0.001f) {
            return Vector3D(x/len, y/len, z/len);
        }
        return Vector3D(0, 0, 0);
    }
    
    float dot(const Vector3D& other) const {
        return x*other.x + y*other.y + z*other.z;
    }
};

struct DroneState {
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;
    Vector3D target;
    float preferred_altitude;
    int formation_group;
    
    DroneState() : preferred_altitude(0), formation_group(0) {}
};

struct DroneWaypoints {
    std::string id;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> time;
    std::vector<float> velocity;
    std::vector<float> heading;
};

struct DroneConfig {
    int num_drones;
    std::string output_file;
    FormationType formation;
    bool enable_collision_avoidance;
    bool enable_swarm_behavior;
    bool enable_traffic_lanes;
    float traffic_lane_width;
    int num_lanes;
};

class CollisionDetector {
private:
    std::unordered_map<int, std::vector<int>> spatial_grid;
    float cell_size;
    int grid_size;
    
public:
    CollisionDetector(float arena_size, float cell_size = MIN_SEPARATION * 2)
        : cell_size(cell_size) {
        grid_size = static_cast<int>(arena_size * 2 / cell_size) + 1;
    }
    
    void clear() {
        spatial_grid.clear();
    }
    
    int getCellIndex(float x, float y) const {
        int grid_x = static_cast<int>((x - X_MIN) / cell_size);
        int grid_y = static_cast<int>((y - Y_MIN) / cell_size);
        return grid_y * grid_size + grid_x;
    }
    
    void addDrone(int drone_id, float x, float y) {
        int cell_idx = getCellIndex(x, y);
        spatial_grid[cell_idx].push_back(drone_id);
    }
    
    std::vector<int> getNearbyDrones(int drone_id, float x, float y, float radius) const {
        std::vector<int> nearby;
        std::set<int> unique_ids;
        
        int center_cell = getCellIndex(x, y);
        
        // Check 3x3 grid around center
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int cell_idx = center_cell + dx + dy * grid_size;
                auto it = spatial_grid.find(cell_idx);
                if (it != spatial_grid.end()) {
                    for (int id : it->second) {
                        if (id != drone_id && unique_ids.find(id) == unique_ids.end()) {
                            unique_ids.insert(id);
                            nearby.push_back(id);
                        }
                    }
                }
            }
        }
        
        return nearby;
    }
};

// Helper functions
template<typename T>
T clamp_value(const T& value, const T& low, const T& high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

float calculateDistance(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

bool isValidPosition(float x, float y, float z) {
    return (x >= X_MIN && x <= X_MAX &&
            y >= Y_MIN && y <= Y_MAX &&
            z >= Z_MIN && z <= Z_MAX);
}

// Reynolds flocking rules
Vector3D calculateSwarmForces(int drone_id, const DroneState& state,
                            const std::vector<DroneState>& all_states,
                            const CollisionDetector& detector) {
    Vector3D cohesion_force(0, 0, 0);
    Vector3D separation_force(0, 0, 0);
    Vector3D alignment_force(0, 0, 0);
    
    int neighbor_count = 0;
    Vector3D center_of_mass(0, 0, 0);
    Vector3D avg_velocity(0, 0, 0);
    
    auto nearby = detector.getNearbyDrones(drone_id, state.position.x, state.position.y, SWARM_RADIUS);
    
    for (int neighbor_id : nearby) {
        if (neighbor_id >= static_cast<int>(all_states.size())) continue;
        
        const DroneState& neighbor = all_states[neighbor_id];
        Vector3D diff = state.position - neighbor.position;
        float distance = diff.length();
        
        if (distance < SWARM_RADIUS && distance > 0.1f) {
            neighbor_count++;
            center_of_mass = center_of_mass + neighbor.position;
            avg_velocity = avg_velocity + neighbor.velocity;
            
            if (distance < PERSONAL_SPACE) {
                float strength = std::min(1.0f, PERSONAL_SPACE / distance - 1.0f);
                separation_force = separation_force + diff.normalized() * strength;
            }
        }
    }
    
    if (neighbor_count > 0) {
        center_of_mass = center_of_mass * (1.0f / neighbor_count);
        Vector3D to_center = center_of_mass - state.position;
        float dist_to_center = to_center.length();
        if (dist_to_center > 0.1f) {
            cohesion_force = to_center.normalized() * std::min(1.0f, dist_to_center / SWARM_RADIUS);
        }
        
        avg_velocity = avg_velocity * (1.0f / neighbor_count);
        alignment_force = (avg_velocity - state.velocity) * 0.1f;
    }
    
    Vector3D total_force = 
        cohesion_force * SWARM_COHESION +
        separation_force * SWARM_SEPARATION +
        alignment_force * SWARM_ALIGNMENT;
    
    float force_mag = total_force.length();
    if (force_mag > MAX_ACCELERATION) {
        total_force = total_force.normalized() * MAX_ACCELERATION;
    }
    
    return total_force;
}

// Collision avoidance
Vector3D calculateAvoidanceForce(int drone_id, const DroneState& state,
                               const std::vector<DroneState>& all_states,
                               const CollisionDetector& detector,
                               float lookahead_time = 2.0f) {
    Vector3D avoidance_force(0, 0, 0);
    
    Vector3D future_pos = state.position + state.velocity * lookahead_time;
    auto nearby = detector.getNearbyDrones(drone_id, future_pos.x, future_pos.y, MIN_SEPARATION * 3);
    
    for (int neighbor_id : nearby) {
        if (neighbor_id >= static_cast<int>(all_states.size())) continue;
        
        const DroneState& neighbor = all_states[neighbor_id];
        Vector3D neighbor_future = neighbor.position + neighbor.velocity * lookahead_time;
        Vector3D diff = future_pos - neighbor_future;
        float distance = diff.length();
        
        if (distance < CRITICAL_SEPARATION && distance > 0.1f) {
            Vector3D rel_velocity = state.velocity - neighbor.velocity;
            float rel_speed = rel_velocity.length();
            
            if (rel_speed > 0.1f) {
                float time_to_collision = distance / rel_speed;
                float strength = std::min(MAX_ACCELERATION * 2.0f, 
                                        (CRITICAL_SEPARATION - distance) / 
                                        std::max(0.1f, time_to_collision));
                avoidance_force = avoidance_force + diff.normalized() * strength;
            } else {
                float strength = (CRITICAL_SEPARATION - distance) / CRITICAL_SEPARATION;
                avoidance_force = avoidance_force + diff.normalized() * strength * MAX_ACCELERATION;
            }
        }
    }
    
    return avoidance_force;
}

// Generate formation positions
std::vector<Vector3D> generateFormationPositions(int num_drones, FormationType formation) {
    std::vector<Vector3D> positions(num_drones);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> alt_dist(Z_MIN + 20, Z_MAX - 20);
    
    switch (formation) {
        case FORMATION_GRID: {
            int grid_size = std::ceil(std::sqrt(num_drones));
            float spacing = std::min(30.0f, (X_MAX - X_MIN) / (grid_size + 1));
            
            for (int i = 0; i < num_drones; ++i) {
                int row = i / grid_size;
                int col = i % grid_size;
                float x = X_MIN + spacing * (col + 1) + (row % 2) * spacing * 0.5f;
                float y = Y_MIN + spacing * (row + 1);
                float z = alt_dist(gen);
                positions[i] = Vector3D(x, y, z);
            }
            break;
        }
        
        case FORMATION_SPIRAL: {
            float center_x = (X_MIN + X_MAX) / 2;
            float center_y = (Y_MIN + Y_MAX) / 2;
            
            for (int i = 0; i < num_drones; ++i) {
                float angle = (2 * M_PI * i) / num_drones;
                float radius = 50.0f + (i * 10.0f / num_drones) * 400.0f;
                float x = center_x + radius * std::cos(angle);
                float y = center_y + radius * std::sin(angle);
                float z = Z_MIN + 50 + (i % 5) * 15.0f;
                positions[i] = Vector3D(x, y, z);
            }
            break;
        }
        
        case FORMATION_V: {
            float center_x = (X_MIN + X_MAX) / 2;
            float center_y = (Y_MIN + Y_MAX) / 2;
            int v_size = std::min(50, num_drones);
            
            for (int i = 0; i < v_size; ++i) {
                int side = (i % 2 == 0) ? 1 : -1;
                int row = i / 2;
                float x = center_x + side * (row * 15.0f);
                float y = center_y + row * 15.0f;
                float z = alt_dist(gen);
                positions[i] = Vector3D(x, y, z);
            }
            
            for (int i = v_size; i < num_drones; ++i) {
                float x = X_MIN + (X_MAX - X_MIN) * (i % 100) / 100.0f;
                float y = Y_MIN + (Y_MAX - Y_MIN) * ((i / 100) % 100) / 100.0f;
                float z = alt_dist(gen);
                positions[i] = Vector3D(x, y, z);
            }
            break;
        }
        
        case FORMATION_RANDOM_WALK: {
            int clusters = std::min(10, num_drones / 10 + 1);
            int drones_per_cluster = num_drones / clusters;
            
            for (int cluster = 0; cluster < clusters; ++cluster) {
                float cluster_x = X_MIN + (X_MAX - X_MIN) * (0.2f + 0.6f * cluster / clusters);
                float cluster_y = Y_MIN + (Y_MAX - Y_MIN) * (0.2f + 0.6f * (cluster % 3) / 3.0f);
                
                for (int i = 0; i < drones_per_cluster && (cluster * drones_per_cluster + i) < num_drones; ++i) {
                    int idx = cluster * drones_per_cluster + i;
                    float angle = 2 * M_PI * (i / static_cast<float>(drones_per_cluster));
                    float radius = 20.0f + (i % 5) * 5.0f;
                    float x = cluster_x + radius * std::cos(angle);
                    float y = cluster_y + radius * std::sin(angle);
                    float z = alt_dist(gen) + (cluster % 3) * 10.0f;
                    positions[idx] = Vector3D(x, y, z);
                }
            }
            
            // Handle remaining drones
            for (int i = clusters * drones_per_cluster; i < num_drones; ++i) {
                float x = X_MIN + 100 + (gen() % 1400);
                float y = Y_MIN + 100 + (gen() % 1400);
                float z = alt_dist(gen);
                positions[i] = Vector3D(x, y, z);
            }
            break;
        }
        
        case FORMATION_RANDOM:
        default: {
            std::uniform_real_distribution<float> x_dist(X_MIN + 50, X_MAX - 50);
            std::uniform_real_distribution<float> y_dist(Y_MIN + 50, Y_MAX - 50);
            
            for (int i = 0; i < num_drones; ++i) {
                int attempts = 0;
                bool placed = false;
                
                while (!placed && attempts < 100) {
                    float x = x_dist(gen);
                    float y = y_dist(gen);
                    float z = alt_dist(gen);
                    
                    placed = true;
                    for (int j = 0; j < i; ++j) {
                        float dx = x - positions[j].x;
                        float dy = y - positions[j].y;
                        float dz = z - positions[j].z;
                        
                        if (dx*dx + dy*dy < MIN_SEPARATION*MIN_SEPARATION &&
                            std::abs(dz) < VERTICAL_BUFFER) {
                            placed = false;
                            break;
                        }
                    }
                    
                    if (placed) {
                        positions[i] = Vector3D(x, y, z);
                    }
                    attempts++;
                }
                
                if (!placed) {
                    // Fallback
                    float x = X_MIN + 100 + (gen() % 1400);
                    float y = Y_MIN + 100 + (gen() % 1400);
                    float z = alt_dist(gen);
                    positions[i] = Vector3D(x, y, z);
                }
            }
            break;
        }
    }
    
    return positions;
}

// Run simulation
std::vector<DroneWaypoints> runSimulation(const DroneConfig& config,
                                        const std::vector<Vector3D>& start_positions,
                                        const std::vector<Vector3D>& target_positions) {
    std::vector<DroneWaypoints> drones(config.num_drones);
    std::vector<DroneState> all_states(config.num_drones);
    
    // Initialize states
    for (int i = 0; i < config.num_drones; ++i) {
        all_states[i].position = start_positions[i];
        all_states[i].target = target_positions[i];
        all_states[i].preferred_altitude = start_positions[i].z;
        all_states[i].velocity = Vector3D(0, 0, 0);
        all_states[i].formation_group = i % 10;
    }
    
    // Main simulation loop
    for (int step = 0; step < NUM_WAYPOINTS; ++step) {
        float current_time = step * TIME_STEP;
        
        // Create collision detector
        CollisionDetector detector(X_MAX - X_MIN);
        for (int i = 0; i < config.num_drones; ++i) {
            detector.addDrone(i, all_states[i].position.x, all_states[i].position.y);
        }
        
        // Update each drone
        for (int i = 0; i < config.num_drones; ++i) {
            DroneState& state = all_states[i];
            
            // Dynamic target update
            if (step % 100 == 0 && step > 0) {
                std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)) + i + step);
                std::uniform_real_distribution<float> x_dist(X_MIN + 100, X_MAX - 100);
                std::uniform_real_distribution<float> y_dist(Y_MIN + 100, Y_MAX - 100);
                state.target = Vector3D(x_dist(gen), y_dist(gen), 
                                       state.preferred_altitude - 20 + (gen() % 41));
            }
            
            // Calculate forces
            Vector3D avoidance_force(0, 0, 0);
            Vector3D swarm_force(0, 0, 0);
            
            if (config.enable_collision_avoidance) {
                avoidance_force = calculateAvoidanceForce(i, state, all_states, detector);
            }
            
            if (config.enable_swarm_behavior) {
                swarm_force = calculateSwarmForces(i, state, all_states, detector);
            }
            
            // Target seeking
            Vector3D to_target = state.target - state.position;
            float dist_to_target = to_target.length();
            Vector3D target_force(0, 0, 0);
            
            if (dist_to_target > 0.1f) {
                float desired_speed = MAX_VELOCITY;
                float slowing_radius = 50.0f;
                
                if (dist_to_target < slowing_radius) {
                    desired_speed = MAX_VELOCITY * (dist_to_target / slowing_radius);
                }
                
                Vector3D desired_velocity = to_target.normalized() * desired_speed;
                target_force = (desired_velocity - state.velocity) * 0.5f;
            }
            
            // Altitude maintenance
            Vector3D altitude_force(0, 0, 0);
            float altitude_error = state.preferred_altitude - state.position.z;
            if (std::abs(altitude_error) > 5.0f) {
                float climb_rate = std::min(MAX_ASCENT_RATE, std::abs(altitude_error) / 2.0f);
                altitude_force.z = (altitude_error > 0 ? climb_rate : -climb_rate);
            }
            
            // Combine forces
            Vector3D total_acceleration = target_force + avoidance_force * 2.0f + 
                                        swarm_force + altitude_force;
            
            // Limit acceleration
            float accel_mag = total_acceleration.length();
            if (accel_mag > MAX_ACCELERATION) {
                total_acceleration = total_acceleration.normalized() * MAX_ACCELERATION;
            }
            
            // Update state
            state.acceleration = total_acceleration;
            state.velocity = state.velocity + state.acceleration * TIME_STEP;
            
            // Limit velocity
            float speed = state.velocity.length();
            if (speed > MAX_VELOCITY) {
                state.velocity = state.velocity.normalized() * MAX_VELOCITY;
            }
            
            // Apply vertical speed limits
            state.velocity.z = clamp_value(state.velocity.z, -MAX_DESCENT_RATE, MAX_ASCENT_RATE);
            
            // Update position
            state.position = state.position + state.velocity * TIME_STEP;
            
            // Apply bounds
            state.position.x = clamp_value(state.position.x, X_MIN + 20, X_MAX - 20);
            state.position.y = clamp_value(state.position.y, Y_MIN + 20, Y_MAX - 20);
            state.position.z = clamp_value(state.position.z, Z_MIN + 5, Z_MAX - 5);
            
            // Initialize waypoint storage if first step
            if (step == 0) {
                drones[i].id = "drone_" + std::to_string(i + 1);
                drones[i].x.reserve(NUM_WAYPOINTS);
                drones[i].y.reserve(NUM_WAYPOINTS);
                drones[i].z.reserve(NUM_WAYPOINTS);
                drones[i].time.reserve(NUM_WAYPOINTS);
                drones[i].velocity.reserve(NUM_WAYPOINTS);
                drones[i].heading.reserve(NUM_WAYPOINTS);
            }
            
            // Store waypoint
            drones[i].time.push_back(current_time);
            drones[i].x.push_back(state.position.x);
            drones[i].y.push_back(state.position.y);
            drones[i].z.push_back(state.position.z);
            drones[i].velocity.push_back(speed);
            
            // Calculate heading
            if (speed > 0.1f) {
                float heading_rad = std::atan2(state.velocity.y, state.velocity.x);
                float heading_deg = heading_rad * 180.0f / M_PI;
                if (heading_deg < 0) heading_deg += 360.0f;
                drones[i].heading.push_back(heading_deg);
            } else {
                drones[i].heading.push_back(drones[i].heading.empty() ? 0.0f : drones[i].heading.back());
            }
        }
        
        // Progress indicator
        if (step % 200 == 0) {
            std::cout << "  Time: " << current_time << "s (" 
                      << (step * 100 / NUM_WAYPOINTS) << "%)" << std::endl;
        }
    }
    
    return drones;
}

// Validate trajectories
bool validateTrajectories(const std::vector<DroneWaypoints>& drones) {
    std::cout << "Validating trajectories..." << std::endl;
    
    int collision_warnings = 0;
    int min_separation_violations = 0;
    int bounds_violations = 0;
    
    // Check every 10th waypoint
    for (int t = 0; t < NUM_WAYPOINTS; t += 10) {
        // Create spatial grid for this time step
        CollisionDetector detector(X_MAX - X_MIN);
        for (size_t i = 0; i < drones.size(); ++i) {
            detector.addDrone(i, drones[i].x[t], drones[i].y[t]);
        }
        
        for (size_t i = 0; i < drones.size(); ++i) {
            // Check bounds
            if (!isValidPosition(drones[i].x[t], drones[i].y[t], drones[i].z[t])) {
                bounds_violations++;
            }
            
            // Check nearby drones
            auto nearby = detector.getNearbyDrones(i, drones[i].x[t], drones[i].y[t], MIN_SEPARATION * 2);
            
            for (int j : nearby) {
                if (j <= static_cast<int>(i)) continue;
                
                float dist = calculateDistance(drones[i].x[t], drones[i].y[t], drones[i].z[t],
                                              drones[j].x[t], drones[j].y[t], drones[j].z[t]);
                
                if (dist < CRITICAL_SEPARATION) {
                    collision_warnings++;
                    if (collision_warnings <= 3) {
                        std::cout << "  Warning: " << drones[i].id << " and " << drones[j].id
                                  << " too close at t=" << (t * TIME_STEP) 
                                  << "s (distance: " << dist << "m)" << std::endl;
                    }
                } else if (dist < MIN_SEPARATION) {
                    min_separation_violations++;
                }
            }
        }
    }
    
    std::cout << "\nValidation Results:" << std::endl;
    std::cout << "  Critical warnings: " << collision_warnings << std::endl;
    std::cout << "  Min separation warnings: " << min_separation_violations << std::endl;
    std::cout << "  Bounds violations: " << bounds_violations << std::endl;
    
    return collision_warnings == 0;
}

// Parse command line arguments
DroneConfig parseArguments(int argc, char* argv[]) {
    DroneConfig config;
    config.num_drones = DEFAULT_NUM_DRONES;
    config.output_file = "drone_waypoints.json";
    config.formation = FORMATION_RANDOM;
    config.enable_collision_avoidance = true;
    config.enable_swarm_behavior = true;
    config.enable_traffic_lanes = false;
    config.traffic_lane_width = 50.0f;
    config.num_lanes = 5;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--drones" && i + 1 < argc) {
            config.num_drones = std::atoi(argv[++i]);
            if (config.num_drones < 1) config.num_drones = 1;
            else if (config.num_drones > MAX_DRONES) config.num_drones = MAX_DRONES;
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--formation" && i + 1 < argc) {
            std::string formation = argv[++i];
            if (formation == "grid") config.formation = FORMATION_GRID;
            else if (formation == "spiral") config.formation = FORMATION_SPIRAL;
            else if (formation == "v") config.formation = FORMATION_V;
            else if (formation == "random_walk") config.formation = FORMATION_RANDOM_WALK;
            else config.formation = FORMATION_RANDOM;
        } else if (arg == "--no-collision-avoidance") {
            config.enable_collision_avoidance = false;
        } else if (arg == "--no-swarm") {
            config.enable_swarm_behavior = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --drones <num>        Number of drones (1-1000, default: " 
                      << DEFAULT_NUM_DRONES << ")" << std::endl;
            std::cout << "  --output <file>       Output JSON file" << std::endl;
            std::cout << "  --formation <type>    random, grid, spiral, v, random_walk" << std::endl;
            std::cout << "  --no-collision-avoidance  Disable collision avoidance" << std::endl;
            std::cout << "  --no-swarm            Disable swarm behavior" << std::endl;
            std::cout << "  --help                Show this help" << std::endl;
            exit(0);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Drone Swarm Waypoint Generator" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    DroneConfig config = parseArguments(argc, argv);
    
    std::cout << "Number of drones: " << config.num_drones << std::endl;
    std::cout << "Time step: " << TIME_STEP << "s, Total time: " << TOTAL_TIME << "s" << std::endl;
    std::cout << "Waypoints per drone: " << NUM_WAYPOINTS << std::endl;
    std::cout << "Collision avoidance: " << (config.enable_collision_avoidance ? "✓" : "✗") << std::endl;
    std::cout << "Swarm behavior: " << (config.enable_swarm_behavior ? "✓" : "✗") << std::endl;
    
    std::cout << "Formation: ";
    switch (config.formation) {
        case FORMATION_GRID: std::cout << "Grid"; break;
        case FORMATION_SPIRAL: std::cout << "Spiral"; break;
        case FORMATION_V: std::cout << "V-formation"; break;
        case FORMATION_RANDOM_WALK: std::cout << "Random Walk"; break;
        default: std::cout << "Random"; break;
    }
    std::cout << std::endl << std::endl;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Generate positions
    std::cout << "Generating initial positions..." << std::endl;
    auto start_positions = generateFormationPositions(config.num_drones, config.formation);
    auto target_positions = generateFormationPositions(config.num_drones, FORMATION_RANDOM);
    
    // Run simulation
    std::cout << "\nGenerating trajectories..." << std::endl;
    auto drones = runSimulation(config, start_positions, target_positions);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nSimulation completed in " << duration.count() << " seconds" << std::endl;
    
    // Validate
    std::cout << "\nValidating trajectories..." << std::endl;
    bool safe = validateTrajectories(drones);
    
    // Create JSON
    json j;
    j["metadata"] = {
        {"num_drones", config.num_drones},
        {"time_step", TIME_STEP},
        {"total_time", TOTAL_TIME},
        {"num_waypoints_per_drone", NUM_WAYPOINTS},
        {"collision_avoidance", config.enable_collision_avoidance},
        {"swarm_behavior", config.enable_swarm_behavior},
        {"formation", config.formation}
    };
    
    json drones_json = json::array();
    for (const auto& drone : drones) {
        json drone_json = {
            {"id", drone.id},
            {"waypoints", json::array()}
        };
        
        for (int i = 0; i < NUM_WAYPOINTS; ++i) {
            drone_json["waypoints"].push_back({
                {"time", drone.time[i]},
                {"x", drone.x[i]},
                {"y", drone.y[i]},
                {"z", drone.z[i]},
                {"velocity", drone.velocity[i]},
                {"heading", drone.heading[i]}
            });
        }
        
        drones_json.push_back(drone_json);
    }
    
    j["drones"] = drones_json;
    
    // Write to file
    std::ofstream file(config.output_file);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
        
        // Calculate statistics
        float total_distance = 0;
        float max_speed = 0;
        float avg_speed = 0;
        
        for (const auto& drone : drones) {
            for (float v : drone.velocity) {
                total_distance += v * TIME_STEP;
                avg_speed += v;
                max_speed = std::max(max_speed, v);
            }
        }
        
        avg_speed /= (drones.size() * NUM_WAYPOINTS);
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Generation Complete!" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Output file: " << config.output_file << std::endl;
        std::cout << "File size: ~" << (j.dump().size() / 1024 / 1024) << " MB" << std::endl;
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Average speed: " << avg_speed << " m/s" << std::endl;
        std::cout << "  Maximum speed: " << max_speed << " m/s" << std::endl;
        std::cout << "  Total distance flown: " << total_distance/1000 << " km" << std::endl;
        
        if (safe) {
            std::cout << "\n✓ Trajectories generated successfully!" << std::endl;
        } else {
            std::cout << "\n⚠ Warnings detected in trajectories" << std::endl;
        }
        
    } else {
        std::cerr << "Error: Could not open file " << config.output_file << std::endl;
        return 1;
    }
    
    return 0;
}