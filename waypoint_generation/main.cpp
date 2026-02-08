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
const int DEFAULT_NUM_DRONES = 49; // number of scheduled drones
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

// Primary drone parameters
const float PRIMARY_DRONE_SPEED_FACTOR = 1.2f;  // Primary drone moves faster
const int PRIMARY_START_TIME_MIN = 50;          // When primary drone appears (seconds)
const int PRIMARY_START_TIME_MAX = 150;         // When primary drone appears (seconds)

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
    bool is_primary_drone;
    
    DroneState() : preferred_altitude(0), formation_group(0), is_primary_drone(false) {}
};

struct DroneWaypoints {
    std::string id;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> time;
    std::vector<float> velocity;
    std::vector<float> heading;
    bool is_primary_drone;
    
    DroneWaypoints() : is_primary_drone(false) {}
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
    bool generate_primary_drone;
    std::string primary_drone_file;
    
    DroneConfig() : 
        num_drones(DEFAULT_NUM_DRONES),
        output_file("drone_waypoints.json"),
        formation(FORMATION_RANDOM),
        enable_collision_avoidance(true),
        enable_swarm_behavior(true),
        enable_traffic_lanes(false),
        traffic_lane_width(50.0f),
        num_lanes(5),
        generate_primary_drone(false),
        primary_drone_file("primary_waypoint.json") {}
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

DroneWaypoints generatePrimaryDroneTrajectory(const std::vector<DroneWaypoints>& regular_drones,
                                               int primary_start_step) {
    std::mt19937 gen(std::random_device{}());
    
    // Primary drone is active from the start (time 0) and stays with swarm
    std::uniform_real_distribution<float> speed_dist(MAX_VELOCITY * 0.4f, MAX_VELOCITY * 0.7f);
    std::uniform_real_distribution<float> small_perturb(-0.2f, 0.2f);
    
    DroneWaypoints primary_drone;
    primary_drone.id = "primary_drone";
    primary_drone.is_primary_drone = true;
    
    // START IN THE MIDDLE OF THE SWARM
    // Find average position of drones at time 0 to start in the swarm
    float avg_x = 0, avg_y = 0, avg_z = 0;
    int count = std::min(10, (int)regular_drones.size());
    for (int i = 0; i < count; ++i) {
        avg_x += regular_drones[i].x[0];
        avg_y += regular_drones[i].y[0];
        avg_z += regular_drones[i].z[0];
    }
    avg_x /= count;
    avg_y /= count;
    avg_z /= count;
    
    // Start near the swarm center
    Vector3D current_position(
        avg_x + small_perturb(gen) * 20.0f,
        avg_y + small_perturb(gen) * 20.0f,
        avg_z + small_perturb(gen) * 5.0f
    );
    
    // Start with normal speed
    float current_speed = speed_dist(gen);
    
    // Choose a nearby drone to initially follow
    int follow_drone = gen() % regular_drones.size();
    Vector3D follow_position(
        regular_drones[follow_drone].x[0],
        regular_drones[follow_drone].y[0],
        regular_drones[follow_drone].z[0]
    );
    
    // Initial velocity - toward the drone we're following
    Vector3D to_follow = follow_position - current_position;
    Vector3D current_velocity = to_follow.normalized() * current_speed;
    
    // Choose 2-3 victim drones that are NEARBY in the swarm
    std::uniform_int_distribution<int> num_victims_dist(2, 3);
    int num_victims = num_victims_dist(gen);
    
    std::vector<int> victim_ids;
    std::vector<int> victim_collision_steps;
    
    // Find drones that are nearby at different times
    for (int i = 0; i < num_victims; ++i) {
        // Look for drones that will be nearby at future times
        int best_victim = -1;
        int best_time = -1;
        float best_distance = 999999.0f;
        
        // Search for potential collision victims
        for (int attempt = 0; attempt < 50; ++attempt) {
            int candidate = gen() % regular_drones.size();
            
            // Try different times between 20-180 seconds
            int collision_time = 200 + (gen() % 1600); // 20-180 seconds
            
            if (collision_time >= NUM_WAYPOINTS - 10) continue;
            
            // Check if this drone will be reasonably close at that time
            float future_x = regular_drones[candidate].x[collision_time];
            float future_y = regular_drones[candidate].y[collision_time];
            float future_z = regular_drones[candidate].z[collision_time];
            
            // Predict where primary will be (rough estimate)
            float pred_x = current_position.x + current_velocity.x * collision_time * TIME_STEP;
            float pred_y = current_position.y + current_velocity.y * collision_time * TIME_STEP;
            
            float distance = std::sqrt(
                (future_x - pred_x) * (future_x - pred_x) +
                (future_y - pred_y) * (future_y - pred_y)
            );
            
            if (distance < 100.0f && distance < best_distance) {
                best_distance = distance;
                best_victim = candidate;
                best_time = collision_time;
            }
        }
        
        if (best_victim != -1) {
            victim_ids.push_back(best_victim);
            victim_collision_steps.push_back(best_time);
        } else {
            // Fallback: random drone and time
            victim_ids.push_back(gen() % regular_drones.size());
            victim_collision_steps.push_back(200 + (gen() % 1600));
        }
    }
    
    // Sort by time
    for (int i = 0; i < victim_collision_steps.size() - 1; ++i) {
        for (int j = i + 1; j < victim_collision_steps.size(); ++j) {
            if (victim_collision_steps[i] > victim_collision_steps[j]) {
                std::swap(victim_ids[i], victim_ids[j]);
                std::swap(victim_collision_steps[i], victim_collision_steps[j]);
            }
        }
    }
    
    // State for following the swarm
    int current_follow_drone = follow_drone;
    int follow_change_counter = 100; // Change followed drone every 10 seconds
    
    // Generate full trajectory - PRIMARY DRONE STAYS WITH SWARM
    for (int step = 0; step < NUM_WAYPOINTS; ++step) {
        float current_time = step * TIME_STEP;
        
        // Check if we're at a collision step
        bool is_collision_step = false;
        int collision_victim_id = -1;
        
        for (int i = 0; i < victim_ids.size(); ++i) {
            if (step == victim_collision_steps[i]) {
                is_collision_step = true;
                collision_victim_id = victim_ids[i];
                break;
            }
        }
        
        if (is_collision_step && collision_victim_id >= 0) {
            // COLLISION - match exact position
            current_position = Vector3D(
                regular_drones[collision_victim_id].x[step],
                regular_drones[collision_victim_id].y[step],
                regular_drones[collision_victim_id].z[step]
            );
            
            // Take on victim's velocity but continue
            if (step > 0) {
                float victim_vx = (regular_drones[collision_victim_id].x[step] - 
                                  regular_drones[collision_victim_id].x[step-1]) / TIME_STEP;
                float victim_vy = (regular_drones[collision_victim_id].y[step] - 
                                  regular_drones[collision_victim_id].y[step-1]) / TIME_STEP;
                
                // Combine with current velocity
                current_velocity = (current_velocity * 0.4f + 
                                   Vector3D(victim_vx, victim_vy, 0) * 0.6f) * 1.1f;
                current_speed = current_velocity.length();
                
                // Random new follow drone after collision
                current_follow_drone = gen() % regular_drones.size();
            }
            
        } else {
            // NORMAL FLIGHT - STAY WITH THE SWARM
            
            // Periodically change which drone we're following
            follow_change_counter--;
            if (follow_change_counter <= 0 || 
                current_follow_drone >= regular_drones.size() ||
                step >= regular_drones[current_follow_drone].x.size()) {
                
                // Find a drone that's nearby to follow
                float min_distance = 999999.0f;
                int closest_drone = current_follow_drone;
                
                for (int i = 0; i < std::min(20, (int)regular_drones.size()); ++i) {
                    int test_drone = (current_follow_drone + i) % regular_drones.size();
                    if (step >= regular_drones[test_drone].x.size()) continue;
                    
                    float dx = regular_drones[test_drone].x[step] - current_position.x;
                    float dy = regular_drones[test_drone].y[step] - current_position.y;
                    float distance = std::sqrt(dx*dx + dy*dy);
                    
                    if (distance < min_distance && distance > 5.0f) {
                        min_distance = distance;
                        closest_drone = test_drone;
                    }
                }
                
                current_follow_drone = closest_drone;
                follow_change_counter = 100 + (gen() % 100); // 10-20 seconds
            }
            
            // Check if approaching a collision soon
            bool approaching_collision = false;
            int next_collision_step = -1;
            int next_victim_id = -1;
            
            for (int i = 0; i < victim_ids.size(); ++i) {
                if (step < victim_collision_steps[i]) {
                    int steps_to_go = victim_collision_steps[i] - step;
                    if (steps_to_go < 50) { // Within 5 seconds
                        approaching_collision = true;
                        next_collision_step = victim_collision_steps[i];
                        next_victim_id = victim_ids[i];
                        break;
                    }
                }
            }
            
            Vector3D target_position;
            float desired_speed = current_speed;
            
            if (approaching_collision) {
                // Steer toward collision point
                target_position = Vector3D(
                    regular_drones[next_victim_id].x[next_collision_step],
                    regular_drones[next_victim_id].y[next_collision_step],
                    regular_drones[next_victim_id].z[next_collision_step]
                );
                
                // Speed up slightly when heading for collision
                desired_speed = std::min(current_speed * 1.2f, MAX_VELOCITY * 0.8f);
            } else {
                // Follow the chosen drone
                if (step < regular_drones[current_follow_drone].x.size()) {
                    target_position = Vector3D(
                        regular_drones[current_follow_drone].x[step],
                        regular_drones[current_follow_drone].y[step],
                        regular_drones[current_follow_drone].z[step]
                    );
                } else {
                    // Fallback: maintain current course
                    target_position = current_position + current_velocity;
                }
                
                // Normal speed variations
                desired_speed = speed_dist(gen);
            }
            
            // Calculate direction to target
            Vector3D to_target = target_position - current_position;
            float distance_to_target = to_target.length();
            
            if (distance_to_target > 0.1f) {
                Vector3D target_direction = to_target.normalized();
                
                // Smooth turning
                Vector3D current_dir = current_velocity.normalized();
                float turn_rate = TURN_RATE * TIME_STEP;
                float dot_product = clamp_value(current_dir.dot(target_direction), -1.0f, 1.0f);
                float angle_diff = std::acos(dot_product);
                
                if (angle_diff > turn_rate) {
                    float t = turn_rate / angle_diff;
                    target_direction = current_dir + (target_direction - current_dir) * t;
                    target_direction = target_direction.normalized();
                }
                
                // Small perturbations for natural movement
                Vector3D perturb(small_perturb(gen), small_perturb(gen), small_perturb(gen) * 0.05f);
                target_direction = (target_direction + perturb * 0.1f).normalized();
                
                // Adjust speed smoothly
                float speed_diff = desired_speed - current_speed;
                float acceleration = clamp_value(speed_diff * 0.8f, 
                                                -MAX_DECELERATION * 0.3f, 
                                                MAX_ACCELERATION * 0.3f);
                current_speed += acceleration * TIME_STEP;
                
                // Update velocity
                current_velocity = target_direction * current_speed;
            }
            
            // Update position
            current_position = current_position + current_velocity * TIME_STEP;
            
            // Small altitude adjustments to match swarm
            if (step % 30 == 0 && current_follow_drone < regular_drones.size() && 
                step < regular_drones[current_follow_drone].z.size()) {
                float target_z = regular_drones[current_follow_drone].z[step];
                float z_diff = target_z - current_position.z;
                if (std::abs(z_diff) > 2.0f) {
                    current_position.z += clamp_value(z_diff * 0.1f, -1.0f, 1.0f);
                }
            }
        }
        
        // Keep within swarm area (but allow some drift)
        current_position.x = clamp_value(current_position.x, X_MIN + 50, X_MAX - 50);
        current_position.y = clamp_value(current_position.y, Y_MIN + 50, Y_MAX - 50);
        current_position.z = clamp_value(current_position.z, Z_MIN + 30, Z_MAX - 30);
        
        // Maintain reasonable speed
        current_speed = current_velocity.length();
        if (current_speed > MAX_VELOCITY * 0.8f) {
            current_velocity = current_velocity.normalized() * MAX_VELOCITY * 0.8f;
            current_speed = MAX_VELOCITY * 0.8f;
        } else if (current_speed < MAX_VELOCITY * 0.3f) {
            current_velocity = current_velocity.normalized() * MAX_VELOCITY * 0.3f;
            current_speed = MAX_VELOCITY * 0.3f;
        }
        
        // Store waypoint
        primary_drone.time.push_back(current_time);
        primary_drone.x.push_back(current_position.x);
        primary_drone.y.push_back(current_position.y);
        primary_drone.z.push_back(current_position.z);
        primary_drone.velocity.push_back(current_speed);
        
        // Calculate heading
        if (current_speed > 0.1f) {
            float heading_rad = std::atan2(current_velocity.y, current_velocity.x);
            float heading_deg = heading_rad * 180.0f / M_PI;
            if (heading_deg < 0) heading_deg += 360.0f;
            primary_drone.heading.push_back(heading_deg);
        } else {
            primary_drone.heading.push_back(primary_drone.heading.empty() ? 0.0f : 
                                           primary_drone.heading.back());
        }
    }
    
    std::cout << "\nPrimary drone behavior (FIXED - stays with swarm):" << std::endl;
    std::cout << "  Starts in swarm center at time 0" << std::endl;
    std::cout << "  Follows other drones in the swarm" << std::endl;
    std::cout << "  Collisions with " << victim_ids.size() << " drones:" << std::endl;
    for (int i = 0; i < victim_ids.size(); ++i) {
        std::cout << "    - " << regular_drones[victim_ids[i]].id 
                  << " at t=" << (victim_collision_steps[i] * TIME_STEP) << "s" << std::endl;
    }
    std::cout << "  Continues flying after each collision" << std::endl;
    
    return primary_drone;
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
        all_states[i].is_primary_drone = false;
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
                drones[i].is_primary_drone = false;
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
            if (!drones[i].is_primary_drone) {
                detector.addDrone(i, drones[i].x[t], drones[i].y[t]);
            }
        }
        
        for (size_t i = 0; i < drones.size(); ++i) {
            if (drones[i].is_primary_drone) continue;
            
            // Check bounds
            if (!isValidPosition(drones[i].x[t], drones[i].y[t], drones[i].z[t])) {
                bounds_violations++;
            }
            
            // Check nearby drones
            auto nearby = detector.getNearbyDrones(i, drones[i].x[t], drones[i].y[t], MIN_SEPARATION * 2);
            
            for (int j : nearby) {
                if (j <= static_cast<int>(i) || drones[j].is_primary_drone) continue;
                
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
        } else if (arg == "--primary-drone") {
            config.generate_primary_drone = true;
        } else if (arg == "--primary-drone-file" && i + 1 < argc) {
            config.primary_drone_file = argv[++i];
            config.generate_primary_drone = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --drones <num>        Number of drones (1-1000, default: " 
                      << DEFAULT_NUM_DRONES << ")" << std::endl;
            std::cout << "  --output <file>       Output JSON file for regular drones" << std::endl;
            std::cout << "  --formation <type>    random, grid, spiral, v, random_walk" << std::endl;
            std::cout << "  --no-collision-avoidance  Disable collision avoidance" << std::endl;
            std::cout << "  --no-swarm            Disable swarm behavior" << std::endl;
            std::cout << "  --primary-drone       Generate a primary drone" << std::endl;
            std::cout << "  --primary-drone-file <file>  Output file for primary drone" << std::endl;
            std::cout << "  --help                Show this help" << std::endl;
            exit(0);
        }
    }
    
    return config;
}

// Save drone waypoints to JSON file
void saveDroneWaypointsToFile(const std::string& filename, const DroneWaypoints& drone) {
    json j;
    
    j["id"] = drone.id;
    j["is_collision_drone"] = drone.is_primary_drone;
    
    json waypoints_json = json::array();
    for (size_t i = 0; i < drone.time.size(); ++i) {
        waypoints_json.push_back({
            {"time", drone.time[i]},
            {"x", drone.x[i]},
            {"y", drone.y[i]},
            {"z", drone.z[i]},
            {"velocity", drone.velocity[i]},
            {"heading", drone.heading[i]}
        });
    }
    
    j["waypoints"] = waypoints_json;
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
        std::cout << "  Saved primary drone to: " << filename << std::endl;
    } else {
        std::cerr << "  Error: Could not open file " << filename << std::endl;
    }
}

// Save all drones to JSON file
void saveAllDronesToFile(const std::string& filename, const std::vector<DroneWaypoints>& drones) {
    json j;
    j["metadata"] = {
        {"num_drones", static_cast<int>(drones.size())},
        {"time_step", TIME_STEP},
        {"total_time", TOTAL_TIME},
        {"num_waypoints_per_drone", NUM_WAYPOINTS}
    };
    
    json drones_json = json::array();
    for (const auto& drone : drones) {
        json drone_json = {
            {"id", drone.id},
            {"is_collision_drone", drone.is_primary_drone},
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
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
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
    std::cout << "Primary drone: " << (config.generate_primary_drone ? "✓" : "✗") << std::endl;
    
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
    
    // Generate primary drone if requested
    DroneWaypoints primary_drone;
    if (config.generate_primary_drone) {
        std::cout << "\nGenerating primary drone..." << std::endl;
        
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> start_step_dist(
            PRIMARY_START_TIME_MIN / TIME_STEP,
            PRIMARY_START_TIME_MAX / TIME_STEP
        );
        
        int primary_start_step = start_step_dist(gen);
        primary_drone = generatePrimaryDroneTrajectory(drones, primary_start_step);
        
        // Save primary drone to separate file
        saveDroneWaypointsToFile(config.primary_drone_file, primary_drone);
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nSimulation completed in " << duration.count() << " seconds" << std::endl;
    
    // Validate (only regular drones)
    std::cout << "\nValidating trajectories..." << std::endl;
    bool safe = validateTrajectories(drones);
    
    // Save all regular drones to file
    saveAllDronesToFile(config.output_file, drones);
    
    // Calculate statistics for regular drones
    float total_distance = 0;
    float max_speed = 0;
    float avg_speed = 0;
    int count = 0;
    
    for (const auto& drone : drones) {
        if (drone.is_primary_drone) continue;
        
        count++;
        for (float v : drone.velocity) {
            total_distance += v * TIME_STEP;
            avg_speed += v;
            max_speed = std::max(max_speed, v);
        }
    }
    
    avg_speed /= (count * NUM_WAYPOINTS);
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Generation Complete!" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Regular drones file: " << config.output_file << std::endl;
    std::cout << "Regular drones count: " << count << std::endl;
    
    if (config.generate_primary_drone) {
        std::cout << "Primary drone file: " << config.primary_drone_file << std::endl;
    }
    
    std::cout << "\nStatistics for regular drones:" << std::endl;
    std::cout << "  Average speed: " << avg_speed << " m/s" << std::endl;
    std::cout << "  Maximum speed: " << max_speed << " m/s" << std::endl;
    std::cout << "  Total distance flown: " << total_distance/1000 << " km" << std::endl;
    
    if (safe) {
        std::cout << "\n✓ Regular drone trajectories generated successfully!" << std::endl;
    } else {
        std::cout << "\n⚠ Warnings detected in regular drone trajectories" << std::endl;
    }
    
    if (config.generate_primary_drone) {
        std::cout << "\n⚠ Primary drone generated - will collide with regular drones!" << std::endl;
    }
    
    return 0;
}