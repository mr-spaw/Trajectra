
// Compile: g++ main.cpp -o uav_sim -lglut -lGLU -lGL -std=c++17 -O3 -pthread

#include <GL/glut.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ============================================================================
// Enhanced Vector3D Class with Full 3D Math Support
// ============================================================================

class Vector3D {
public:
    float x, y, z;
    
    Vector3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    Vector3D operator+(const Vector3D& v) const {
        return Vector3D(x + v.x, y + v.y, z + v.z);
    }
    
    Vector3D operator-(const Vector3D& v) const {
        return Vector3D(x - v.x, y - v.y, z - v.z);
    }
    
    Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    Vector3D operator/(float scalar) const {
        if (scalar != 0) return Vector3D(x / scalar, y / scalar, z / scalar);
        return *this;
    }
    
    Vector3D& operator+=(const Vector3D& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    
    float dot(const Vector3D& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    Vector3D cross(const Vector3D& v) const {
        return Vector3D(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    float length() const {
        return sqrt(x * x + y * y + z * z);
    }
    
    float lengthSquared() const {
        return x * x + y * y + z * z;
    }
    
    Vector3D normalized() const {
        float len = length();
        if (len > 0.0001f) return *this / len;
        return Vector3D(0, 0, 0);
    }
    
    float distance(const Vector3D& v) const {
        return (*this - v).length();
    }
    
    float distanceSquared(const Vector3D& v) const {
        return (*this - v).lengthSquared();
    }
    
    Vector3D lerp(const Vector3D& target, float t) const {
        return *this + (target - *this) * t;
    }
    
    json toJson() const {
        return json::object({
            {"x", x},
            {"y", y},
            {"z", z}
        });
    }
    
    static Vector3D fromJson(const json& j) {
        return Vector3D(
            j.value("x", 0.0f),
            j.value("y", 0.0f),
            j.value("z", 0.0f)
        );
    }
};

// ============================================================================
// Configuration Constants
// ============================================================================
namespace Config {

    // Adjust based on your JSON bounds
    const float WORLD_SIZE = 1600.0f;           // Based on JSON bounds (-400 to 400)
    const float GRID_SIZE = 1600.0f;
    const float SAFETY_BUFFER = 5.0f;
    const int MAX_DRONES = 1000;
    const int SPATIAL_GRID_DIVISIONS = 40;
    const float LOD_DISTANCE_NEAR = 400.0f;
    const float LOD_DISTANCE_MID = 1000.0f;
    const float LOD_DISTANCE_FAR = 2000.0f;
    
    // Primary drone configuration - ENHANCED FOR VISIBILITY
    static const Vector3D PRIMARY_DRONE_COLOR() { return Vector3D(0.0f, 1.0f, 0.0f); }  
    const float PRIMARY_DRONE_SIZE = 15.0f;  
}

// ============================================================================
// JSON-based Collision Data Loader
// ============================================================================
struct CollisionData {
    std::string primary_drone_id;
    std::string conflicting_drone_id;
    Vector3D primary_location;
    Vector3D conflict_location;
    Vector3D location;
    float time;
    float distance;
    std::string severity;
    
    CollisionData() 
        : time(0.0f), distance(0.0f), severity("LOW") {}
    
    json toJson() const {
        return json::object({
            {"primary_drone", primary_drone_id},
            {"conflicting_drone", conflicting_drone_id},
            {"primary_location", primary_location.toJson()},
            {"conflict_location", conflict_location.toJson()},
            {"location", location.toJson()},
            {"time", time},
            {"distance", distance},
            {"severity", severity}
        });
    }
    
    static CollisionData fromJson(const json& j) {
        CollisionData data;
        data.primary_drone_id = j.value("primary_drone", "");
        data.conflicting_drone_id = j.value("conflicting_drone", "");
        data.time = j.value("time", 0.0f);
        data.distance = j.value("distance", 0.0f);
        data.severity = j.value("severity", "LOW");
        
        if (j.contains("primary_location")) {
            data.primary_location = Vector3D::fromJson(j["primary_location"]);
        }
        
        if (j.contains("conflict_location")) {
            data.conflict_location = Vector3D::fromJson(j["conflict_location"]);
        }
        
        if (j.contains("location")) {
            data.location = Vector3D::fromJson(j["location"]);
        }
        
        return data;
    }
};

class JSONCollisionLoader {
private:
    std::vector<CollisionData> collisions;
    mutable std::mutex data_mutex;
    std::string analysis_file_path;
    
public:
    JSONCollisionLoader(const std::string& file_path = "") : analysis_file_path(file_path) {
        if (!file_path.empty()) {
            loadFromJSON(file_path);
        }
    }
    
    bool loadFromJSON(const std::string& filename) {
        analysis_file_path = filename;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open analysis JSON file: " << filename << std::endl;
            return false;
        }
        
        try {
            json root;
            file >> root;
            
            std::lock_guard<std::mutex> lock(data_mutex);
            collisions.clear();
            
            if (root.contains("metadata")) {
                json metadata = root["metadata"];
                std::cout << "Collision Analysis Metadata:" << std::endl;
                std::cout << "  Total conflicts: " << metadata.value("total_conflicts", 0) << std::endl;
                std::cout << "  Total drones: " << metadata.value("total_drones", 0) << std::endl;
                std::cout << "  Safety radius: " << metadata.value("safety_radius", 10.0f) << "m" << std::endl;
            }
            
            if (root.contains("conflicts")) {
                json json_conflicts = root["conflicts"];
                for (const auto& conflict : json_conflicts) {
                    CollisionData data = CollisionData::fromJson(conflict);
                    collisions.push_back(data);
                }
                
                std::cout << "Loaded " << collisions.size() << " collision entries from JSON." << std::endl;
                
                auto it = std::unique(collisions.begin(), collisions.end(),
                    [](const CollisionData& a, const CollisionData& b) {
                        return a.primary_drone_id == b.primary_drone_id &&
                               a.conflicting_drone_id == b.conflicting_drone_id &&
                               std::abs(a.time - b.time) < 0.1f;
                    });
                collisions.erase(it, collisions.end());
                
                std::cout << "After deduplication: " << collisions.size() << " unique collision entries." << std::endl;
                return true;
            }
            return false;
        }
        catch (const json::exception& e) {
            std::cerr << "Error: JSON parsing error: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
    }
    
    void reload() {
        if (!analysis_file_path.empty()) {
            loadFromJSON(analysis_file_path);
        }
    }
    
    std::vector<CollisionData> getCollisionsAtTime(float time, float tolerance = 0.5f) const {
        std::lock_guard<std::mutex> lock(data_mutex);
        std::vector<CollisionData> result;
        
        for (const auto& collision : collisions) {
            if (std::abs(collision.time - time) <= tolerance) {
                result.push_back(collision);
            }
        }
        return result;
    }
    
    std::vector<CollisionData> getAllCollisions() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return collisions;
    }
    
    std::vector<CollisionData> getCollisionsByDrone(int drone_id) const {
        std::lock_guard<std::mutex> lock(data_mutex);
        std::vector<CollisionData> result;
        std::string drone_str = "drone_" + std::to_string(drone_id);
        
        for (const auto& collision : collisions) {
            if (collision.conflicting_drone_id == drone_str) {
                result.push_back(collision);
            }
        }
        return result;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(data_mutex);
        collisions.clear();
    }
    
    void printStatus() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        std::cout << "[Collision Loader] Total collisions: " << collisions.size() << std::endl;
        
        if (!collisions.empty()) {
            std::cout << "Time range: " << collisions.front().time << "s to " 
                      << collisions.back().time << "s" << std::endl;
            
            std::map<std::string, int> severity_count;
            for (const auto& c : collisions) {
                severity_count[c.severity]++;
            }
            
            std::cout << "By severity: ";
            for (const auto& [severity, count] : severity_count) {
                std::cout << severity << ": " << count << " ";
            }
            std::cout << std::endl;
        }
    }
};

// ============================================================================
// Physics Engine for Realistic Drone Movement
// ============================================================================
class DronePhysics {
public:
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;
    
    float maxSpeed;
    float maxAcceleration;
    float maxVerticalSpeed;
    float mass;
    float drag;
    
    DronePhysics() 
        : maxSpeed(25.0f), maxAcceleration(8.0f), 
          maxVerticalSpeed(10.0f), mass(1.5f), drag(0.1f) {}
    
    void applyForce(const Vector3D& force) {
        acceleration += force / mass;
    }
    
    void update(float deltaTime) {
        Vector3D dragForce = velocity * (-drag);
        applyForce(dragForce);
        
        float accMag = acceleration.length();
        if (accMag > maxAcceleration) {
            acceleration = acceleration.normalized() * maxAcceleration;
        }
        
        velocity += acceleration * deltaTime;
        
        Vector3D horizVel(velocity.x, velocity.y, 0);
        float horizSpeed = horizVel.length();
        if (horizSpeed > maxSpeed) {
            float scale = maxSpeed / horizSpeed;
            velocity.x *= scale;
            velocity.y *= scale;
        }
        
        if (fabs(velocity.z) > maxVerticalSpeed) {
            velocity.z = (velocity.z > 0 ? 1 : -1) * maxVerticalSpeed;
        }
        
        position += velocity * deltaTime;
        acceleration = Vector3D(0, 0, 0);
    }
    
    void seekTarget(const Vector3D& target, float arrivalRadius = 5.0f) {
        Vector3D desired = target - position;
        float distance = desired.length();
        
        if (distance < 0.01f) return;
        
        desired = desired.normalized();
        
        if (distance < arrivalRadius) {
            float speed = maxSpeed * (distance / arrivalRadius);
            desired = desired * speed;
        } else {
            desired = desired * maxSpeed;
        }
        
        Vector3D steer = desired - velocity;
        float steerMag = steer.length();
        if (steerMag > maxAcceleration) {
            steer = steer.normalized() * maxAcceleration;
        }
        
        applyForce(steer * mass);
    }
};

// ============================================================================
// Waypoint and Trajectory with Time Management
// ============================================================================
class Waypoint {
public:
    Vector3D position;
    float timestamp;
    float arrivalRadius;
    
    Waypoint(Vector3D pos = Vector3D(), float time = 0, float radius = 5.0f) 
        : position(pos), timestamp(time), arrivalRadius(radius) {}
};

class Trajectory {
private:
    std::vector<Waypoint> waypoints;
    int currentWaypointIndex;
    
public:
    Trajectory() : currentWaypointIndex(0) {}
    
    void addWaypoint(const Waypoint& wp) {
        waypoints.push_back(wp);
    }
    
    const Waypoint* getCurrentWaypoint() const {
        if (currentWaypointIndex < waypoints.size()) {
            return &waypoints[currentWaypointIndex];
        }
        return nullptr;
    }
    
    bool advanceWaypoint() {
        if (currentWaypointIndex < waypoints.size() - 1) {
            currentWaypointIndex++;
            return true;
        }
        return false;
    }
    
    void reset() {
        currentWaypointIndex = 0;
    }
    
    float getTotalDuration() const {
        if (waypoints.empty()) return 0.0f;
        return waypoints.back().timestamp;
    }
    
    const std::vector<Waypoint>& getWaypoints() const {
        return waypoints;
    }
    
    int getCurrentWaypointIndex() const {
        return currentWaypointIndex;
    }
    
    float getProgress() const {
        if (waypoints.empty()) return 1.0f;
        return (float)currentWaypointIndex / (float)waypoints.size();
    }
    
    Vector3D predictPosition(float futureTime) const {
        if (waypoints.empty()) return Vector3D();
        
        float totalDuration = getTotalDuration();
        if (totalDuration > 0) {
            futureTime = fmod(futureTime, totalDuration);
        }
        
        for (size_t i = 0; i < waypoints.size() - 1; i++) {
            if (futureTime >= waypoints[i].timestamp && 
                futureTime <= waypoints[i+1].timestamp) {
                float t = (futureTime - waypoints[i].timestamp) / 
                         (waypoints[i+1].timestamp - waypoints[i].timestamp);
                return waypoints[i].position.lerp(waypoints[i+1].position, t);
            }
        }
        
        if (futureTime >= waypoints.back().timestamp) {
            return waypoints.back().position;
        }
        return waypoints[0].position;
    }
    
    bool loopTrajectory() {
        if (waypoints.empty()) return false;
        currentWaypointIndex = 0;
        return true;
    }
    
    const Waypoint* getWaypointAtTime(float time) const {
        if (waypoints.empty()) return nullptr;
        
        int low = 0;
        int high = waypoints.size() - 1;
        
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (waypoints[mid].timestamp <= time && 
                (mid == waypoints.size() - 1 || waypoints[mid + 1].timestamp > time)) {
                return &waypoints[mid];
            }
            if (waypoints[mid].timestamp < time) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return &waypoints[0];
    }
    
    Vector3D getPositionAtTime(float time) const {
        if (waypoints.empty()) return Vector3D();
        
        float totalDuration = getTotalDuration();
        if (totalDuration > 0) {
            time = fmod(time, totalDuration);
        }
        
        for (size_t i = 0; i < waypoints.size() - 1; i++) {
            if (time >= waypoints[i].timestamp && 
                time <= waypoints[i+1].timestamp) {
                float t = (time - waypoints[i].timestamp) / 
                         (waypoints[i+1].timestamp - waypoints[i].timestamp);
                return waypoints[i].position.lerp(waypoints[i+1].position, t);
            }
        }
        
        return waypoints.back().position;
    }
};

// ============================================================================
// Drone with Physics and Visual Effects
// ============================================================================
class Drone {
private:
    int id;
    std::string idString;
    Vector3D color;
    Vector3D originalColor;
    Trajectory trajectory;
    DronePhysics physics;
    float currentTime;
    bool isActive;
    float size;
    float rotorAngle;
    float bodyTilt;
    
    // Trail effect - ENHANCED for primary drone
    std::vector<Vector3D> trail;
    std::vector<float> trailTimes;
    int maxTrailLength;
    int trailUpdateCounter;
    
    // For primary drone special effects
    float pulseEffect;
    float searchlightAngle;
    
    // For JSON-based trajectories
    bool usePreciseTrajectory;
    
    // For primary drone
    bool isPrimaryDrone;
    
public:
    Drone(int id, Vector3D color = Vector3D(1, 0, 0), bool primary = false)
        : id(id), color(color), originalColor(color), currentTime(0), 
          isActive(true), size(primary ? Config::PRIMARY_DRONE_SIZE : 2.0f), 
          rotorAngle(0), bodyTilt(0), 
          maxTrailLength(primary ? 500 : 50),
          trailUpdateCounter(0), pulseEffect(0.0f), searchlightAngle(0.0f),
          usePreciseTrajectory(false), isPrimaryDrone(primary) {
        
        idString = primary ? "PRIMARY_DRONE" : ("UAV-" + std::to_string(id));
        if (primary) {
            originalColor = Vector3D(0.0f, 1.0f, 0.0f);
            color = Vector3D(0.0f, 1.0f, 0.0f);
            
            // Initialize trail with some points at starting position
            if (!trajectory.getWaypoints().empty()) {
                Vector3D startPos = trajectory.getWaypoints()[0].position;
                for (int i = 0; i < 20; i++) {
                    trail.push_back(startPos);
                    trailTimes.push_back(0.0f);
                }
            }
        }
    }
    
    void setTrajectory(const Trajectory& traj, bool precise = false) {
        trajectory = traj;
        usePreciseTrajectory = precise;
        if (!trajectory.getWaypoints().empty()) {
            physics.position = trajectory.getWaypoints()[0].position;
        }
    }
    
    void update(float deltaTime) {
        if (!isActive) return;
        
        // Update simulation time
        currentTime += deltaTime;
        
        // Update pulse effect for primary drone
        if (isPrimaryDrone) {
            pulseEffect += deltaTime * 5.0f;
            searchlightAngle += deltaTime * 30.0f;
        }
        
        // Wrap time at 200 seconds for precise trajectory mode
        if (usePreciseTrajectory) {
            float totalDuration = trajectory.getTotalDuration();
            if (totalDuration > 0) {
                if (currentTime >= totalDuration) {
                    currentTime = 0.0f;
                    // Reset trail when restarting
                    trail.clear();
                    trailTimes.clear();
                }
            }
        }
        
        rotorAngle += deltaTime * 1000.0f;
        if (rotorAngle > 360.0f) rotorAngle -= 360.0f;
        
        if (usePreciseTrajectory) {
            // Use precise trajectory from waypoints
            Vector3D targetPos = trajectory.getPositionAtTime(currentTime);
            
            // Calculate velocity from trajectory for smooth movement
            float nextTime = currentTime + 0.1f;
            if (nextTime >= trajectory.getTotalDuration()) {
                nextTime = 0.0f;
            }
            Vector3D nextPos = trajectory.getPositionAtTime(nextTime);
            Vector3D desiredVel = (nextPos - targetPos) * 10.0f;
            
            Vector3D desired = targetPos - physics.position;
            float distance = desired.length();
            
            if (distance > 0.1f) {
                physics.velocity = physics.velocity * 0.7f + desiredVel * 0.3f;
                
                float speed = physics.velocity.length();
                if (speed > 25.0f) {
                    physics.velocity = physics.velocity.normalized() * 25.0f;
                }
                
                physics.position += physics.velocity * deltaTime;
                
                Vector3D horizVel(physics.velocity.x, physics.velocity.y, 0);
                float horizSpeed = horizVel.length();
                bodyTilt = std::min(horizSpeed * 2.0f, 25.0f);
            } else {
                physics.position = targetPos;
            }
        } else {
            // Use physics-based navigation
            const Waypoint* target = trajectory.getCurrentWaypoint();
            if (target) {
                physics.seekTarget(target->position, target->arrivalRadius);
                physics.update(deltaTime);
                
                Vector3D horizVel(physics.velocity.x, physics.velocity.y, 0);
                float speed = horizVel.length();
                bodyTilt = std::min(speed * 2.0f, 25.0f);
                
                if (physics.position.distance(target->position) < target->arrivalRadius) {
                    if (!trajectory.advanceWaypoint()) {
                        trajectory.loopTrajectory();
                    }
                }
            }
        }
 
        if (isPrimaryDrone) {
            // Primary drone: ALWAYS add trail point every frame
            trail.push_back(physics.position);
            trailTimes.push_back(currentTime);
            
            // Keep trail at max length
            if (trail.size() > maxTrailLength) {
                trail.erase(trail.begin());
                trailTimes.erase(trailTimes.begin());
            }
            
            // Also add "breadcrumbs" every 0.5 seconds
            static float breadcrumbTimer = 0.0f;
            breadcrumbTimer += deltaTime;
            if (breadcrumbTimer > 0.5f) {
                // Add an extra visible point
                trail.push_back(physics.position);
                trailTimes.push_back(currentTime);
                breadcrumbTimer = 0.0f;
            }
        } else {
            // Regular drones: update trail less frequently
            trailUpdateCounter++;
            if (trailUpdateCounter >= 3) {
                trail.push_back(physics.position);
                trailTimes.push_back(currentTime);
                if (trail.size() > maxTrailLength) {
                    trail.erase(trail.begin());
                    trailTimes.erase(trailTimes.begin());
                }
                trailUpdateCounter = 0;
            }
        }
    }
    
    void reset() {
        currentTime = 0;
        isActive = true;
        trajectory.reset();
        if (!trajectory.getWaypoints().empty()) {
            physics.position = trajectory.getWaypoints()[0].position;
            physics.velocity = Vector3D(0, 0, 0);
            physics.acceleration = Vector3D(0, 0, 0);
        }
        trail.clear();
        trailTimes.clear();
        rotorAngle = 0;
        bodyTilt = 0;
        trailUpdateCounter = 0;
        pulseEffect = 0.0f;
        searchlightAngle = 0.0f;
    }
    
    int getId() const { return id; }
    std::string getIdString() const { return idString; }
    Vector3D getPosition() const { return physics.position; }
    Vector3D getVelocity() const { return physics.velocity; }
    Vector3D getColor() const { return color; }
    bool getIsActive() const { return isActive; }
    float getSize() const { return size; }
    float getCurrentTime() const { return currentTime; }
    float getRotorAngle() const { return rotorAngle; }
    float getBodyTilt() const { return bodyTilt; }
    const std::vector<Vector3D>& getTrail() const { return trail; }
    const std::vector<float>& getTrailTimes() const { return trailTimes; }
    const Trajectory& getTrajectory() const { return trajectory; }
    bool getIsPrimary() const { return isPrimaryDrone; }
    float getPulseEffect() const { return pulseEffect; }
    float getSearchlightAngle() const { return searchlightAngle; }
    
    void setColor(Vector3D newColor) { 
        if (!isPrimaryDrone) {
            color = newColor; 
        }
    }
    void resetColor() { 
        if (isPrimaryDrone) {
            color = Vector3D(0.0f, 1.0f, 0.0f);
        } else {
            color = originalColor; 
        }
    }
    
    Vector3D predictPosition(float futureTime) const {
        if (usePreciseTrajectory) {
            return trajectory.getPositionAtTime(currentTime + futureTime);
        }
        return trajectory.predictPosition(currentTime + futureTime);
    }
    
    float getSpeed() const {
        return physics.velocity.length();
    }
    
    float getAltitude() const {
        return physics.position.z;
    }
};

// ============================================================================
// Display List Cache
// ============================================================================
class DisplayListCache {
private:
    GLuint droneFullDetail;
    GLuint droneMediumDetail;
    GLuint droneLowDetail;
    GLuint rotorList;
    GLuint primaryDroneList;
    bool initialized;
    
    void createPrimaryDrone() {
    primaryDroneList = glGenLists(1);
    glNewList(primaryDroneList, GL_COMPILE);
    
    float size = Config::PRIMARY_DRONE_SIZE * 0.5f;
    
    // Simple sphere body
    glColor3f(0.0f, 1.0f, 0.0f);
    glutSolidSphere(size, 12, 12);
    
    // Simple wireframe
    glColor3f(0.1f, 0.5f, 0.1f);
    glLineWidth(1.5f);
    glutWireSphere(size * 1.05f, 8, 8);
    glLineWidth(1.0f);
    
    glEndList();
}
    
    void createFullDetailDrone() {
        droneFullDetail = glGenLists(1);
        glNewList(droneFullDetail, GL_COMPILE);
        
        float size = 2.0f;
        
        glPushMatrix();
        glScalef(size, size * 0.8f, size * 0.5f);
        glutSolidSphere(0.7f, 16, 16);
        glPopMatrix();
        
        glColor3f(0.2f, 0.2f, 0.25f);
        glLineWidth(1.5f);
        glPushMatrix();
        glScalef(size, size * 0.8f, size * 0.5f);
        glutWireSphere(0.72f, 12, 12);
        glPopMatrix();
        glLineWidth(1.0f);
        
        float armLength = size * 1.5f;
        float armThickness = size * 0.15f;
        
        for (int i = 0; i < 4; i++) {
            glPushMatrix();
            
            float posX = (i % 2 == 0) ? armLength/2 : -armLength/2;
            float posY = (i < 2) ? armLength/2 : -armLength/2;
            
            glTranslatef(posX, posY, 0);
            
            if (i < 2) {
                glRotatef(90, 0, 0, 1);
            }
            
            glColor3f(0.15f, 0.15f, 0.2f);
            glBegin(GL_QUAD_STRIP);
            for (int j = 0; j <= 8; j++) {
                float t = j / 8.0f;
                float x = armLength/2 - t * armLength;
                float r = armThickness * (1.0f - t * 0.3f);
                
                glVertex3f(x, -r, -r);
                glVertex3f(x, -r, r);
                glVertex3f(x, r, -r);
                glVertex3f(x, r, r);
            }
            glEnd();
            
            glPopMatrix();
        }
        
        glEndList();
    }
    
    void createRotorModel() {
        rotorList = glGenLists(1);
        glNewList(rotorList, GL_COMPILE);
        
        float rotorRadius = 0.8f;
        float bladeThickness = 0.05f;
        
        glColor3f(0.9f, 0.9f, 0.95f);
        
        glutSolidSphere(0.15f, 8, 8);
        
        for (int i = 0; i < 2; i++) {
            glPushMatrix();
            glRotatef(i * 180.0f, 0, 0, 1);
            
            glBegin(GL_QUADS);
            for (int seg = 0; seg < 3; seg++) {
                float t1 = seg / 3.0f;
                float t2 = (seg + 1) / 3.0f;
                
                float r1 = 0.15f + t1 * (rotorRadius - 0.15f);
                float r2 = 0.15f + t2 * (rotorRadius - 0.15f);
                float z1 = sin(t1 * M_PI * 0.5f) * 0.03f;
                float z2 = sin(t2 * M_PI * 0.5f) * 0.03f;
                
                glVertex3f(r1, -bladeThickness, z1);
                glVertex3f(r1, bladeThickness, z1);
                glVertex3f(r2, bladeThickness, z2);
                glVertex3f(r2, -bladeThickness, z2);
            }
            glEnd();
            
            glPopMatrix();
        }
        
        glEndList();
    }
    
    void createMediumDetailDrone() {
        droneMediumDetail = glGenLists(1);
        glNewList(droneMediumDetail, GL_COMPILE);
        
        float size = 2.0f;
        
        glPushMatrix();
        glScalef(size, size * 0.8f, size * 0.5f);
        glutSolidSphere(0.7f, 12, 12);
        glPopMatrix();
        
        float armLength = size * 1.5f;
        glColor3f(0.15f, 0.15f, 0.2f);
        
        glBegin(GL_LINES);
        glVertex3f(armLength/2, armLength/2, 0);
        glVertex3f(-armLength/2, armLength/2, 0);
        glVertex3f(armLength/2, -armLength/2, 0);
        glVertex3f(-armLength/2, -armLength/2, 0);
        glVertex3f(armLength/2, armLength/2, 0);
        glVertex3f(armLength/2, -armLength/2, 0);
        glVertex3f(-armLength/2, armLength/2, 0);
        glVertex3f(-armLength/2, -armLength/2, 0);
        glEnd();
        
        glEndList();
    }
    
    void createLowDetailDrone() {
        droneLowDetail = glGenLists(1);
        glNewList(droneLowDetail, GL_COMPILE);
        
        glPushMatrix();
        glScalef(2.0f, 1.6f, 1.0f);
        glutSolidCube(1.0f);
        glPopMatrix();
        
        glEndList();
    }
    
public:
    DisplayListCache() : initialized(false) {}
    
    void initialize() {
        if (initialized) return;
        
        createPrimaryDrone();
        createFullDetailDrone();
        createRotorModel();
        createMediumDetailDrone();
        createLowDetailDrone();
        
        initialized = true;
    }
    
    void cleanup() {
        if (!initialized) return;
        
        glDeleteLists(droneFullDetail, 1);
        glDeleteLists(droneMediumDetail, 1);
        glDeleteLists(droneLowDetail, 1);
        glDeleteLists(rotorList, 1);
        glDeleteLists(primaryDroneList, 1);
        
        initialized = false;
    }
    
    GLuint getFullDetail() const { return droneFullDetail; }
    GLuint getMediumDetail() const { return droneMediumDetail; }
    GLuint getLowDetail() const { return droneLowDetail; }
    GLuint getRotorModel() const { return rotorList; }
    GLuint getPrimaryDrone() const { return primaryDroneList; }
    
    ~DisplayListCache() {
        cleanup();
    }
};

// ============================================================================
// OpenGL Renderer 
// ============================================================================
class OpenGLRenderer {
private:
    float gridSize;
    float axisLength;
    bool showGrid;
    bool showTrajectories;
    bool showConflicts;
    bool showTrails;
    bool showHUD;
    bool showDroneLabels;
    float cameraAngleX;
    float cameraAngleY;
    float cameraDistance;
    Vector3D cameraTarget;
    DisplayListCache displayCache;
    
    // Performance monitoring
    int frameCount;
    float fpsTimer;
    float currentFPS;
    
    // Camera inertia
    float zoomVelocity;
    float rotVelocityX;
    float rotVelocityY;
    
    // For HUD
    int currentScenario;
    float timeScale;
    std::string currentTrajectoryFile;
    std::string currentAnalysisFile;
    
    // Drawing functions
    void drawText(float x, float y, const std::string& text, void* font = GLUT_BITMAP_HELVETICA_12) {
        glRasterPos2f(x, y);
        for (char c : text) {
            glutBitmapCharacter(font, c);
        }
    }
    
    void drawText3D(float x, float y, float z, const std::string& text) {
        glRasterPos3f(x, y, z);
        for (char c : text) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c);
        }
    }
    
    void drawGrid() {
    glDisable(GL_LIGHTING);
    glLineWidth(1.0f);
    
    float step = gridSize / 20.0f;  // Major grid lines
    
    // Draw main grid (darker lines)
    glColor3f(0.3f, 0.3f, 0.35f);
    glBegin(GL_LINES);
    for (float x = -gridSize; x <= gridSize; x += step) {
        glVertex3f(x, -gridSize, 0.1f);
        glVertex3f(x, gridSize, 0.1f);
        glVertex3f(-gridSize, x, 0.1f);
        glVertex3f(gridSize, x, 0.1f);
    }
    glEnd();
    
    // Draw finer grid lines
    glColor3f(0.2f, 0.2f, 0.25f);
    glBegin(GL_LINES);
    for (float x = -gridSize; x <= gridSize; x += step/4.0f) {
        // Only draw if not a major line
        if (fmod(x, step) != 0) {
            glVertex3f(x, -gridSize, 0.05f);
            glVertex3f(x, gridSize, 0.05f);
            glVertex3f(-gridSize, x, 0.05f);
            glVertex3f(gridSize, x, 0.05f);
        }
    }
    glEnd();
    
    // Draw thicker coordinate lines
    glLineWidth(2.0f);
    glColor3f(0.4f, 0.4f, 0.45f);
    glBegin(GL_LINES);
    
    // X-axis lines
    for (float x = -gridSize; x <= gridSize; x += step * 2) {
        glVertex3f(x, -gridSize, 0.15f);
        glVertex3f(x, gridSize, 0.15f);
        glVertex3f(-gridSize, x, 0.15f);
        glVertex3f(gridSize, x, 0.15f);
    }
    glEnd();
    
    // Draw coordinate labels (optional)
    if (showGrid) {
        glColor3f(0.6f, 0.6f, 0.7f);
        for (float x = -gridSize; x <= gridSize; x += step * 2) {
            if (x != 0) {
                std::ostringstream label;
                label << std::fixed << std::setprecision(0) << x;
                
                // X-axis labels
                drawText3D(x, -gridSize - step/2, 0.2f, label.str());
                drawText3D(x, gridSize + step/2, 0.2f, label.str());
                
                // Y-axis labels
                drawText3D(-gridSize - step/2, x, 0.2f, label.str());
                drawText3D(gridSize + step/2, x, 0.2f, label.str());
            }
        }
        
        // Origin label
        glColor3f(0.8f, 0.8f, 1.0f);
        drawText3D(-step/2, -step/2, 0.2f, "0");
    }
    
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}
    
    void drawAxes() {
    glDisable(GL_LIGHTING);
    glLineWidth(4.0f);
    
    // X-axis (Red)
    glColor3f(0.8f, 0.2f, 0.2f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(axisLength, 0, 0);
    glEnd();
    
    // Y-axis (Green)
    glColor3f(0.2f, 0.8f, 0.2f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, axisLength, 0);
    glEnd();
    
    // Z-axis (Blue)
    glColor3f(0.2f, 0.2f, 0.8f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, axisLength);
    glEnd();
    
    // Arrowheads
    float arrowSize = axisLength * 0.05f;
    
    // X-axis arrow
    glColor3f(0.8f, 0.2f, 0.2f);
    glBegin(GL_TRIANGLES);
    glVertex3f(axisLength, 0, 0);
    glVertex3f(axisLength - arrowSize, arrowSize, 0);
    glVertex3f(axisLength - arrowSize, -arrowSize, 0);
    glEnd();
    
    // Y-axis arrow
    glColor3f(0.2f, 0.8f, 0.2f);
    glBegin(GL_TRIANGLES);
    glVertex3f(0, axisLength, 0);
    glVertex3f(arrowSize, axisLength - arrowSize, 0);
    glVertex3f(-arrowSize, axisLength - arrowSize, 0);
    glEnd();
    
    // Z-axis arrow
    glColor3f(0.2f, 0.2f, 0.8f);
    glBegin(GL_TRIANGLES);
    glVertex3f(0, 0, axisLength);
    glVertex3f(arrowSize, 0, axisLength - arrowSize);
    glVertex3f(-arrowSize, 0, axisLength - arrowSize);
    glEnd();
    
    // Axis labels
    glColor3f(1.0f, 1.0f, 1.0f);
    drawText3D(axisLength + arrowSize, -arrowSize, 0, "X");
    drawText3D(-arrowSize, axisLength + arrowSize, 0, "Y");
    drawText3D(-arrowSize, 0, axisLength + arrowSize, "Z");
    
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}    

    enum LODLevel { LOD_HIGH, LOD_MEDIUM, LOD_LOW, LOD_POINT };
    LODLevel getLODLevel(float distance) const {
        if (distance < Config::LOD_DISTANCE_NEAR) return LOD_HIGH;
        if (distance < Config::LOD_DISTANCE_MID) return LOD_MEDIUM;
        if (distance < Config::LOD_DISTANCE_FAR) return LOD_LOW;
        return LOD_POINT;
    }
    
    // Drone rendering
    void drawDrone(const Drone& drone, const Vector3D& cameraPos) {
        if (drone.getIsPrimary()) {
            drawPrimaryDroneEnhanced(drone, cameraPos);
        } else {
            drawRegularDrone(drone, cameraPos);
        }
    }
    
    void drawRegularDrone(const Drone& drone, const Vector3D& cameraPos) {
        Vector3D pos = drone.getPosition();
        Vector3D color = drone.getColor();
        float distance = pos.distance(cameraPos);
        LODLevel lod = getLODLevel(distance);
        
        if (lod == LOD_POINT) {
            glPointSize(3.0f);
            glDisable(GL_LIGHTING);
            glEnable(GL_POINT_SMOOTH);
            glColor3f(color.x, color.y, color.z);
            glBegin(GL_POINTS);
            glVertex3f(pos.x, pos.y, pos.z);
            glEnd();
            glEnable(GL_LIGHTING);
            return;
        }
        
        glPushMatrix();
        glTranslatef(pos.x, pos.y, pos.z);
        
        if (lod == LOD_HIGH) {
            Vector3D vel = drone.getVelocity();
            if (vel.length() > 0.1f) {
                Vector3D forward = vel.normalized();
                float angle = atan2(forward.y, forward.x) * 180.0f / M_PI;
                glRotatef(angle, 0, 0, 1);
                glRotatef(drone.getBodyTilt(), 0, 1, 0);
            }
        }
        
        GLfloat matAmbient[] = {color.x * 0.3f, color.y * 0.3f, color.z * 0.3f, 1.0f};
        GLfloat matDiffuse[] = {color.x, color.y, color.z, 1.0f};
        GLfloat matSpecular[] = {0.7f, 0.7f, 0.7f, 1.0f};
        GLfloat matShininess[] = {50.0f};
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, matAmbient);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiffuse);
        glMaterialfv(GL_FRONT, GL_SPECULAR, matSpecular);
        glMaterialfv(GL_FRONT, GL_SHININESS, matShininess);
        
        switch (lod) {
            case LOD_HIGH:
                glCallList(displayCache.getFullDetail());
                
                if (distance < Config::LOD_DISTANCE_NEAR / 2) {
                    float rotorAngle = drone.getRotorAngle();
                    float size = drone.getSize();
                    float armLength = size * 1.5f;
                    
                    float rotorPositions[4][2] = {
                        {armLength, armLength},
                        {-armLength, armLength},
                        {armLength, -armLength},
                        {-armLength, -armLength}
                    };
                    
                    for (int i = 0; i < 4; i++) {
                        glPushMatrix();
                        glTranslatef(rotorPositions[i][0], rotorPositions[i][1], size/4);
                        glRotatef(rotorAngle + i * 90, 0, 0, 1);
                        glScalef(0.8f, 0.8f, 0.8f);
                        glCallList(displayCache.getRotorModel());
                        glPopMatrix();
                    }
                }
                break;
                
            case LOD_MEDIUM:
                glCallList(displayCache.getMediumDetail());
                break;
                
            case LOD_LOW:
                glCallList(displayCache.getLowDetail());
                break;
                
            default:
                break;
        }
        
        GLfloat defaultAmbient[] = {0.2f, 0.2f, 0.2f, 1.0f};
        GLfloat defaultDiffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
        glMaterialfv(GL_FRONT, GL_AMBIENT, defaultAmbient);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, defaultDiffuse);
        
        glPopMatrix();
    }
    
    void drawPrimaryDroneEnhanced(const Drone& drone, const Vector3D& cameraPos) {
    Vector3D pos = drone.getPosition();
    
    glPushMatrix();
    glTranslatef(pos.x, pos.y, pos.z);
    

    float pulse = 0.3f * sin(drone.getPulseEffect()) + 0.7f;
    
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    
    // Single subtle glow layer
    glColor4f(0.0f, 0.8f, 0.0f, 0.15f);
    glutSolidSphere(30.0f, 16, 16);
    
    // Pulsing core (smaller)
    glColor4f(0.0f, 1.0f, 0.0f, 0.3f * pulse);
    glutSolidSphere(20.0f, 16, 16);
    
    glEnable(GL_LIGHTING);

    GLfloat matAmbient[] = {0.0f, 0.6f, 0.0f, 1.0f};
    GLfloat matDiffuse[] = {0.0f, 1.0f, 0.0f, 1.0f};
    GLfloat matSpecular[] = {0.5f, 0.5f, 0.5f, 1.0f};
    GLfloat matShininess[] = {50.0f};
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, matAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, matSpecular);
    glMaterialfv(GL_FRONT, GL_SHININESS, matShininess);
    
    // Main body - simple sphere
    float bodySize = Config::PRIMARY_DRONE_SIZE * 0.5f;
    glColor3f(0.0f, 0.9f, 0.0f);
    glutSolidSphere(bodySize, 16, 16);
    
    // Simple wireframe for structure visibility
    glDisable(GL_LIGHTING);
    glColor3f(0.1f, 0.5f, 0.1f);
    glLineWidth(2.0f);
    glutWireSphere(bodySize * 1.05f, 8, 8);
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
    
    // Simple X-frame arms (4 lines)
    float armLength = bodySize * 1.5f;
    glDisable(GL_LIGHTING);
    glColor3f(0.2f, 0.7f, 0.2f);
    glLineWidth(3.0f);
    
    glBegin(GL_LINES);
    for (int i = 0; i < 4; i++) {
        float angle = 45.0f + i * 90.0f;
        float radians = angle * M_PI / 180.0f;
        float x = cos(radians) * armLength;
        float y = sin(radians) * armLength;
        
        glVertex3f(0, 0, 0);
        glVertex3f(x, y, 0);
    }
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
    
    // Simple rotors (small discs)
    float rotorAngle = drone.getRotorAngle();
    float rotorSize = bodySize * 0.6f;
    
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    
    for (int i = 0; i < 4; i++) {
        glPushMatrix();
        
        float angle = 45.0f + i * 90.0f;
        float radians = angle * M_PI / 180.0f;
        
        // Position at end of arm
        float x = cos(radians) * armLength;
        float y = sin(radians) * armLength;
        
        glTranslatef(x, y, 0);
        glRotatef(rotorAngle + i * 90.0f, 0, 0, 1);
        
        // Simple rotor disc
        glColor4f(0.8f, 0.9f, 1.0f, 0.4f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0, 0, 0);
        for (int j = 0; j <= 12; j++) {
            float angle = 2.0f * M_PI * j / 12;
            glVertex3f(cos(angle) * rotorSize, sin(angle) * rotorSize, 0);
        }
        glEnd();
        
        glPopMatrix();
    }

    glPushMatrix();
    glRotatef(drone.getSearchlightAngle(), 0, 0, 1);
    
    // Simple downward beam
    glColor4f(0.0f, 0.7f, 0.0f, 0.1f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, 0, -bodySize);
    
    float beamLength = 100.0f;
    float beamWidth = 25.0f;
    for (int i = 0; i <= 8; i++) {
        float angle = 2.0f * M_PI * i / 8;
        float x = cos(angle) * beamWidth;
        float y = sin(angle) * beamWidth;
        glVertex3f(x, y, -beamLength);
    }
    glEnd();
    
    glPopMatrix();
    
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    
    glPopMatrix();
    
    drawPrimaryDroneLabel(drone);
}
    
    void drawPrimaryDroneSearchlight(const Drone& drone) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        glPushMatrix();
        glRotatef(drone.getSearchlightAngle(), 0, 0, 1);
        
        // Main beam cone
        glColor4f(0.0f, 1.0f, 0.0f, 0.15f);
        
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0, 0, 0);
        
        int segments = 16;
        float beamLength = 150.0f;
        float beamWidth = 40.0f;
        for (int i = 0; i <= segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = cos(angle) * beamWidth;
            float y = sin(angle) * beamWidth;
            glVertex3f(x, y, -beamLength);
        }
        glEnd();
        
        // Ground spotlight
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0.0f, 1.0f, 0.0f, 0.1f);
        glVertex3f(0, 0, -beamLength);
        
        glColor4f(0.0f, 1.0f, 0.0f, 0.01f);
        float radius = 100.0f;
        for (int i = 0; i <= segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = cos(angle) * radius;
            float y = sin(angle) * radius;
            glVertex3f(x, y, -beamLength);
        }
        glEnd();
        
        glPopMatrix();
        
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
   void drawPrimaryDroneLabel(const Drone& drone) {
    if (!showDroneLabels) return;
    
    Vector3D pos = drone.getPosition();
    
    glDisable(GL_LIGHTING);
    

    glColor3f(0.0f, 1.0f, 0.0f);
    std::ostringstream label;
    label << "PRIMARY";
    drawText3D(pos.x - 10, pos.y, pos.z + 25, label.str());
    
    glEnable(GL_LIGHTING);
}
    
    void drawPrimaryDroneVelocityVector(const Drone& drone) {
        if (drone.getSpeed() < 0.1f) return;
        
        Vector3D pos = drone.getPosition();
        Vector3D vel = drone.getVelocity().normalized() * 30.0f;
        
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        glLineWidth(6.0f);
        glColor4f(1.0f, 1.0f, 0.0f, 0.9f);
        glBegin(GL_LINES);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(pos.x + vel.x, pos.y + vel.y, pos.z + vel.z);
        glEnd();
        
        glPushMatrix();
        glTranslatef(pos.x + vel.x, pos.y + vel.y, pos.z + vel.z);
        
        Vector3D forward = vel.normalized();
        float angle = atan2(forward.y, forward.x) * 180.0f / M_PI;
        glRotatef(angle, 0, 0, 1);
        
        glColor4f(1.0f, 0.8f, 0.0f, 1.0f);
        glBegin(GL_TRIANGLES);
        glVertex3f(0, 0, 0);
        glVertex3f(-5.0f, 2.0f, 0);
        glVertex3f(-5.0f, -2.0f, 0);
        glEnd();
        
        glPopMatrix();
        
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
    void drawPrimaryDroneCompass(const Drone& drone) {
        Vector3D pos = drone.getPosition();
        Vector3D vel = drone.getVelocity();
        
        if (vel.length() < 0.1f) return;
        
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        float compassRadius = 25.0f;
        float compassZ = pos.z + 80.0f;
        
        glPushMatrix();
        glTranslatef(pos.x, pos.y, compassZ);
        
        glColor4f(0.0f, 0.5f, 0.0f, 0.6f);
        glLineWidth(3.0f);
        glBegin(GL_LINE_LOOP);
        int segments = 32;
        for (int i = 0; i < segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = cos(angle) * compassRadius;
            float y = sin(angle) * compassRadius;
            glVertex3f(x, y, 0);
        }
        glEnd();
        
        float angle = atan2(vel.y, vel.x);
        float arrowX = cos(angle) * compassRadius;
        float arrowY = sin(angle) * compassRadius;
        
        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glBegin(GL_TRIANGLES);
        glVertex3f(arrowX, arrowY, 0);
        glVertex3f(arrowX * 0.7f + arrowY * 0.3f, arrowY * 0.7f - arrowX * 0.3f, 0);
        glVertex3f(arrowX * 0.7f - arrowY * 0.3f, arrowY * 0.7f + arrowX * 0.3f, 0);
        glEnd();
        
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_TRIANGLES);
        glVertex3f(0, compassRadius, 0);
        glVertex3f(-3.0f, compassRadius - 5.0f, 0);
        glVertex3f(3.0f, compassRadius - 5.0f, 0);
        glEnd();
        
        glColor3f(1.0f, 0.0f, 0.0f);
        drawText3D(-1.5f, compassRadius - 10.0f, 0.1f, "N");
        
        glPopMatrix();
        
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
    void drawPulseRing(float radius, float pulse, const Vector3D& color) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        glColor4f(color.x, color.y, color.z, pulse * 0.3f);
        glLineWidth(3.0f * pulse);
        
        glBegin(GL_LINE_LOOP);
        int segments = 24;
        for (int i = 0; i < segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = cos(angle) * radius;
            float y = sin(angle) * radius;
            glVertex3f(x, y, 0);
        }
        glEnd();
        
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
    void drawTrail(const Drone& drone, const Vector3D& cameraPos) {
        if (!showTrails) return;
        
        if (drone.getIsPrimary()) {
            drawPrimaryDroneTrail(drone);
        } else {
            drawRegularDroneTrail(drone, cameraPos);
        }
    }
    
    void drawPrimaryDroneTrail(const Drone& drone) {
    const auto& trail = drone.getTrail();
    if (trail.size() < 2) return;
    
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    
    // ===== THINNER TRAIL LINE =====
    glLineWidth(2.5f);
    glBegin(GL_LINE_STRIP);
    
    // Simple gradient trail
    for (size_t i = 0; i < trail.size(); i++) {
        float t = (float)i / trail.size();
        float alpha = 0.2f + t * 0.3f;  // Fade out older points
        
        glColor4f(0.0f, 0.8f, 0.3f, alpha);
        glVertex3f(trail[i].x, trail[i].y, trail[i].z);
    }
    glEnd();
    
    // ===== OCCASIONAL BREADCRUMBS =====
    glPointSize(8.0f);  // Reduced from 15.0f
    glBegin(GL_POINTS);
    
    for (size_t i = 0; i < trail.size(); i += 50) {  // Less frequent (was 20)
        float t = (float)i / trail.size();
        float alpha = 0.5f + t * 0.3f;
        
        glColor4f(0.2f, 1.0f, 0.3f, alpha);
        glVertex3f(trail[i].x, trail[i].y, trail[i].z);
    }
    glEnd();
    
    // ===== TIME MARKERS (ONLY KEY POINTS) =====
    if (trail.size() > 100) {
        glPointSize(12.0f);  // Reduced from 20.0f
        glBegin(GL_POINTS);
        
        // Only mark every 200 points (was 100)
        for (size_t i = 0; i < trail.size(); i += 200) {
            glColor4f(1.0f, 1.0f, 0.0f, 0.8f);
            glVertex3f(trail[i].x, trail[i].y, trail[i].z);
        }
        glEnd();
    }
    
    // ===== THIN SHADOW ON GROUND =====
    glLineWidth(3.0f);  // Reduced from 8.0f
    glBegin(GL_LINE_STRIP);
    glColor4f(0.0f, 0.4f, 0.0f, 0.1f);
    
    // Sample fewer points for shadow
    for (size_t i = 0; i < trail.size(); i += 10) {
        glVertex3f(trail[i].x, trail[i].y, 0.1f);
    }
    glEnd();
    
    glLineWidth(1.0f);
    glPointSize(1.0f);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}
    
    void drawRegularDroneTrail(const Drone& drone, const Vector3D& cameraPos) {
        float distance = drone.getPosition().distance(cameraPos);
        if (distance > Config::LOD_DISTANCE_MID) return;
        
        const auto& trail = drone.getTrail();
        if (trail.size() < 2) return;
        
        Vector3D color = drone.getColor();
        
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glLineWidth(1.5f);
        
        glBegin(GL_LINE_STRIP);
        for (size_t i = 0; i < trail.size(); i++) {
            float alpha = pow((float)i / trail.size(), 2.0f);
            glColor4f(color.x, color.y, color.z, alpha * 0.6f);
            glVertex3f(trail[i].x, trail[i].y, trail[i].z);
        }
        glEnd();
        
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
    void drawTrajectory(const Drone& drone, const Vector3D& cameraPos) {
        if (!showTrajectories) return;
        
        if (drone.getIsPrimary()) {
            drawPrimaryDroneTrajectory(drone);
        } else {
            float distance = drone.getPosition().distance(cameraPos);
            if (distance > Config::LOD_DISTANCE_MID) return;
            
            const Trajectory& traj = drone.getTrajectory();
            const auto& waypoints = traj.getWaypoints();
            if (waypoints.size() < 2) return;
            
            glDisable(GL_LIGHTING);
            glColor4f(0.5f, 0.5f, 0.5f, 0.3f);
            glLineWidth(1.0f);
            
            glBegin(GL_LINE_STRIP);
            for (const auto& wp : waypoints) {
                glVertex3f(wp.position.x, wp.position.y, wp.position.z);
            }
            glEnd();
            
            glEnable(GL_LIGHTING);
        }
    }
    
    void drawPrimaryDroneTrajectory(const Drone& drone) {
        const Trajectory& traj = drone.getTrajectory();
        const auto& waypoints = traj.getWaypoints();
        if (waypoints.size() < 2) return;
        
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        // Future trajectory line
        glColor4f(0.0f, 0.8f, 1.0f, 0.6f);
        glLineWidth(3.0f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(3, 0xAAAA);
        
        glBegin(GL_LINE_STRIP);
        
        float currentTime = drone.getCurrentTime();
        for (float t = 0; t <= 20.0f; t += 2.0f) {
            Vector3D futurePos = traj.getPositionAtTime(currentTime + t);
            glVertex3f(futurePos.x, futurePos.y, futurePos.z);
        }
        glEnd();
        
        glDisable(GL_LINE_STIPPLE);
        
        // Waypoint markers
        int currentIdx = traj.getCurrentWaypointIndex();
        for (size_t i = 0; i < waypoints.size(); i++) {
            const auto& wp = waypoints[i];
            
            glPushMatrix();
            glTranslatef(wp.position.x, wp.position.y, wp.position.z);
            
            if ((int)i == currentIdx) {
                glColor4f(1.0f, 1.0f, 0.0f, 0.9f);
                glutSolidSphere(8.0f, 16, 16);
                
                glColor3f(1.0f, 1.0f, 1.0f);
                std::ostringstream wpText;
                wpText << "WP " << i;
                drawText3D(-5, -5, 10, wpText.str());
            } else if ((int)i > currentIdx) {
                glColor4f(0.0f, 1.0f, 0.0f, 0.7f);
                glutWireSphere(5.0f, 12, 12);
            } else {
                glColor4f(0.5f, 0.5f, 0.5f, 0.4f);
                glutWireSphere(3.0f, 8, 8);
            }
            
            glPopMatrix();
        }
        
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
    
    void drawCollisionZone(const CollisionData& collision, float currentTime) {
        Vector3D pos = collision.location;
        
        if (std::abs(collision.time - currentTime) > 2.0f) return;
        
        glPushMatrix();
        glTranslatef(pos.x, pos.y, pos.z);
        
        float red, green, blue, alpha;
        
        if (collision.severity == "HIGH") {
            red = 1.0f;
            green = 0.0f;
            blue = 0.0f;
            alpha = 0.3f;
        } else if (collision.severity == "MEDIUM") {
            red = 1.0f;
            green = 0.5f;
            blue = 0.0f;
            alpha = 0.25f;
        } else {
            red = 1.0f;
            green = 1.0f;
            blue = 0.0f;
            alpha = 0.2f;
        }
        
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        
        glColor4f(red, green, blue, alpha * 0.5f);
        glutSolidSphere(20.0f, 16, 16);
        
        glColor4f(red, green, blue, alpha);
        glutSolidSphere(15.0f, 16, 16);
        
        static float pulse = 0.0f;
        pulse += 0.15f;
        float pulseSize = 18.0f + sin(pulse) * 3.0f;
        
        glColor4f(red, green, blue, alpha * 0.4f);
        glutWireSphere(pulseSize, 12, 12);
        
        glColor4f(1.0f, 1.0f, 1.0f, 0.6f);
        glutWireSphere(16.0f, 8, 8);
        
        glBegin(GL_LINES);
        glColor4f(1.0f, 0.5f, 0.0f, 0.8f);
        glVertex3f(collision.primary_location.x - pos.x, 
                  collision.primary_location.y - pos.y, 
                  collision.primary_location.z - pos.z);
        glVertex3f(collision.conflict_location.x - pos.x, 
                  collision.conflict_location.y - pos.y, 
                  collision.conflict_location.z - pos.z);
        glEnd();
        
        glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
        std::ostringstream distText;
        distText << std::fixed << std::setprecision(1) << collision.distance << "m";
        drawText3D(-5, -5, 10, distText.str());
        
        std::ostringstream timeText;
        timeText << "t=" << std::fixed << std::setprecision(1) << collision.time << "s";
        drawText3D(-5, -5, 15, timeText.str());
        
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
        glPopMatrix();
    }
    
    void drawHUD(const std::vector<std::unique_ptr<Drone>>& drones, 
             const JSONCollisionLoader& collisionLoader, float simTime) {
    // Switch to 2D orthographic projection for HUD
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 1280, 0, 720);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Background gradient
    glBegin(GL_QUADS);
    glColor4f(0.0f, 0.05f, 0.1f, 0.85f);
    glVertex2f(0, 720);
    glVertex2f(1280, 720);
    glColor4f(0.0f, 0.1f, 0.15f, 0.85f);
    glVertex2f(1280, 680);
    glVertex2f(0, 680);
    glEnd();
    
    // Top border
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor4f(0.0f, 0.8f, 1.0f, 0.8f);
    glVertex2f(0, 680);
    glVertex2f(1280, 680);
    glEnd();
    glLineWidth(1.0f);
    
    // Statistics
    int activeDrones = 0;
    int primaryDrones = 0;
    float avgAltitude = 0;
    
    for (const auto& drone : drones) {
        if (drone->getIsActive()) {
            activeDrones++;
            avgAltitude += drone->getAltitude();
            if (drone->getIsPrimary()) primaryDrones++;
        }
    }
    
    if (activeDrones > 0) avgAltitude /= activeDrones;
    
    glColor3f(0.8f, 0.9f, 1.0f);
    
    // Left side: Time and drones
    std::ostringstream topLeft;
    topLeft << "🕒 TIME: " << std::fixed << std::setprecision(1) << simTime << "s";
    topLeft << "   🚁 DRONES: " << activeDrones << "/" << drones.size();
    topLeft << " (Primary: " << primaryDrones << ")";
    topLeft << "   📊 FPS: " << std::setprecision(0) << currentFPS;
    drawText(20, 700, topLeft.str(), GLUT_BITMAP_HELVETICA_12);
    
    // Right side: Collision status
    auto collisions = collisionLoader.getAllCollisions();
    auto currentCollisions = collisionLoader.getCollisionsAtTime(simTime, 1.0f);
    
    std::string statusText;
    Vector3D statusColor(0.8f, 0.8f, 0.8f);
    
    if (!currentCollisions.empty()) {
        statusText = "⚠ " + std::to_string(currentCollisions.size()) + " ACTIVE COLLISIONS";
        statusColor = Vector3D(1.0f, 0.3f, 0.3f);
    } else if (!collisions.empty()) {
        int upcoming = 0;
        for (const auto& c : collisions) {
            if (c.time > simTime && c.time <= simTime + 10.0f) {
                upcoming++;
            }
        }
        if (upcoming > 0) {
            statusText = "🔶 " + std::to_string(upcoming) + " UPCOMING COLLISIONS";
            statusColor = Vector3D(1.0f, 0.8f, 0.3f);
        } else {
            statusText = "✅ ALL CLEAR";
            statusColor = Vector3D(0.3f, 1.0f, 0.3f);
        }
    } else {
        statusText = "📁 NO COLLISION DATA LOADED";
        statusColor = Vector3D(0.7f, 0.7f, 0.9f);
    }
    
    glColor3f(statusColor.x, statusColor.y, statusColor.z);
    drawText(1000, 700, statusText, GLUT_BITMAP_HELVETICA_12);
    
    // ============================================
    // 2. PRIMARY DRONE STATUS PANEL 
    // ============================================
    const Drone* primaryDrone = nullptr;
    for (const auto& drone : drones) {
        if (drone->getIsPrimary()) {
            primaryDrone = drone.get();
            break;
        }
    }
    
    if (primaryDrone) {
        float panelX = 20;
        float panelY = 680 - 140;  // Positioned below top bar
        float panelWidth = 380;
        float panelHeight = 130;
        
        // Panel background with border
        glColor4f(0.0f, 0.1f, 0.05f, 0.85f);
        glBegin(GL_QUADS);
        glVertex2f(panelX, panelY);
        glVertex2f(panelX + panelWidth, panelY);
        glVertex2f(panelX + panelWidth, panelY + panelHeight);
        glVertex2f(panelX, panelY + panelHeight);
        glEnd();
        
        // Panel border
        glLineWidth(2.0f);
        glColor4f(0.0f, 0.8f, 0.3f, 0.9f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(panelX, panelY);
        glVertex2f(panelX + panelWidth, panelY);
        glVertex2f(panelX + panelWidth, panelY + panelHeight);
        glVertex2f(panelX, panelY + panelHeight);
        glEnd();
        glLineWidth(1.0f);
        
        // Title
        glColor3f(0.0f, 1.0f, 0.5f);
        drawText(panelX + 15, panelY + panelHeight - 25, 
                "🚁 PRIMARY DRONE STATUS", GLUT_BITMAP_HELVETICA_12);
        
        // Separator line
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glColor4f(0.0f, 0.6f, 0.2f, 0.6f);
        glVertex2f(panelX + 10, panelY + panelHeight - 35);
        glVertex2f(panelX + panelWidth - 10, panelY + panelHeight - 35);
        glEnd();
        
        // Position info
        Vector3D pos = primaryDrone->getPosition();
        Vector3D vel = primaryDrone->getVelocity();
        
        float textY = panelY + panelHeight - 55;
        
        std::ostringstream info;
        glColor3f(0.9f, 0.9f, 0.9f);
        
        info.str(""); info.clear();
        info << "📍 Position: X=" << std::fixed << std::setprecision(1) << pos.x 
             << "m  Y=" << pos.y << "m  Z=" << pos.z << "m";
        drawText(panelX + 20, textY, info.str(), GLUT_BITMAP_HELVETICA_10);
        
        info.str(""); info.clear();
        info << "⚡ Velocity: " << std::setprecision(1) << primaryDrone->getSpeed() 
             << " m/s  (" << std::setprecision(1) << vel.x << ", " 
             << std::setprecision(1) << vel.y << ", " 
             << std::setprecision(1) << vel.z << ")";
        drawText(panelX + 20, textY - 15, info.str(), GLUT_BITMAP_HELVETICA_10);
        
        info.str(""); info.clear();
        info << "🕒 Trajectory Time: " << std::setprecision(1) 
             << primaryDrone->getCurrentTime() << "s";
        drawText(panelX + 20, textY - 30, info.str(), GLUT_BITMAP_HELVETICA_10);
        
        info.str(""); info.clear();
        info << "🎯 Next Waypoint: " << (primaryDrone->getTrajectory().getCurrentWaypointIndex() + 1) 
             << "/" << primaryDrone->getTrajectory().getWaypoints().size();
        drawText(panelX + 20, textY - 45, info.str(), GLUT_BITMAP_HELVETICA_10);
    }
    
    // ============================================
    // 3. COLLISION ALERT PANEL (RIGHT SIDE)
    // ============================================
    if (!currentCollisions.empty()) {
        float alertX = 1280 - 320;
        float alertY = 680 - 100;
        float alertWidth = 300;
        float alertHeight = 90;
        
        // Alert background (pulsing red)
        float pulse = 0.5f * sin(simTime * 10.0f) + 0.5f;
        glColor4f(0.5f + pulse * 0.3f, 0.1f, 0.1f, 0.9f);
        glBegin(GL_QUADS);
        glVertex2f(alertX, alertY);
        glVertex2f(alertX + alertWidth, alertY);
        glVertex2f(alertX + alertWidth, alertY + alertHeight);
        glVertex2f(alertX, alertY + alertHeight);
        glEnd();
        
        // Alert border
        glLineWidth(3.0f);
        glColor4f(1.0f, 0.3f, 0.3f, 0.9f + pulse * 0.1f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(alertX, alertY);
        glVertex2f(alertX + alertWidth, alertY);
        glVertex2f(alertX + alertWidth, alertY + alertHeight);
        glVertex2f(alertX, alertY + alertHeight);
        glEnd();
        glLineWidth(1.0f);
        
        // Alert text
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(alertX + 10, alertY + alertHeight - 25, 
                "⚠ COLLISION ALERT!", GLUT_BITMAP_HELVETICA_12);
        
        const auto& collision = currentCollisions[0];
        glColor3f(1.0f, 0.9f, 0.9f);
        
        std::ostringstream alertText;
        alertText << "Primary vs " << collision.conflicting_drone_id;
        drawText(alertX + 15, alertY + alertHeight - 45, 
                alertText.str(), GLUT_BITMAP_HELVETICA_10);
        
        alertText.str(""); alertText.clear();
        alertText << "Distance: " << std::fixed << std::setprecision(1) 
                 << collision.distance << "m";
        alertText << "  Severity: " << collision.severity;
        drawText(alertX + 15, alertY + alertHeight - 60, 
                alertText.str(), GLUT_BITMAP_HELVETICA_10);
    }
    
    // ============================================
    // 4. FILE INFO PANEL (BOTTOM LEFT)
    // ============================================
    float infoX = 20;
    float infoY = 40;
    float infoWidth = 500;
    float infoHeight = 50;
    
    glColor4f(0.0f, 0.05f, 0.1f, 0.7f);
    glBegin(GL_QUADS);
    glVertex2f(infoX, infoY);
    glVertex2f(infoX + infoWidth, infoY);
    glVertex2f(infoX + infoWidth, infoY + infoHeight);
    glVertex2f(infoX, infoY + infoHeight);
    glEnd();
    
    glColor3f(0.7f, 0.8f, 1.0f);
    std::ostringstream fileInfo;
    fileInfo << "📁 Trajectory: " << currentTrajectoryFile;
    drawText(infoX + 10, infoY + 35, fileInfo.str(), GLUT_BITMAP_HELVETICA_10);
    
    if (!currentAnalysisFile.empty()) {
        glColor3f(0.7f, 1.0f, 0.8f);
        std::ostringstream analysisInfo;
        analysisInfo << "📊 Analysis: " << currentAnalysisFile 
                    << " (" << collisions.size() << " collisions)";
        drawText(infoX + 10, infoY + 20, analysisInfo.str(), GLUT_BITMAP_HELVETICA_10);
    }
    
    glColor3f(1.0f, 1.0f, 0.8f);
    std::ostringstream speedInfo;
    speedInfo << "⚡ Simulation Speed: " << std::fixed << std::setprecision(1) 
             << timeScale << "x";
    drawText(infoX + 10, infoY + 5, speedInfo.str(), GLUT_BITMAP_HELVETICA_10);
    
    // ============================================
    // 5. CONTROLS HINT (BOTTOM RIGHT)
    // ============================================
    float controlsX = 1280 - 400;
    float controlsY = 20;
    
    glColor4f(0.0f, 0.05f, 0.1f, 0.7f);
    glBegin(GL_QUADS);
    glVertex2f(controlsX, controlsY);
    glVertex2f(1280, controlsY);
    glVertex2f(1280, controlsY + 35);
    glVertex2f(controlsX, controlsY + 35);
    glEnd();
    
    glColor4f(0.8f, 0.9f, 1.0f, 0.9f);
    std::string line1 = "[H] HUD  [G] Grid  [T] Trails  [C] Collisions  [L] Trajectories";
    std::string line2 = "[1-4] Speed  [SPACE] Pause  [R] Reset  [V] Camera  [S] Status";
    
    drawText(controlsX + 10, controlsY + 25, line1, GLUT_BITMAP_HELVETICA_10);
    drawText(controlsX + 10, controlsY + 10, line2, GLUT_BITMAP_HELVETICA_10);
    
    // ============================================
    // 6. SIMULATION STATS (TOP RIGHT)
    // ============================================
    float statsX = 1280 - 250;
    float statsY = 680 - 30;
    
    glColor3f(0.6f, 0.8f, 1.0f);
    std::ostringstream stats;
    stats << "📈 Avg Altitude: " << std::fixed << std::setprecision(1) 
          << avgAltitude << "m";
    drawText(statsX, statsY, stats.str(), GLUT_BITMAP_HELVETICA_10);
    
    // Restore OpenGL state
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}
    
public:
    OpenGLRenderer() 
        : gridSize(Config::GRID_SIZE), axisLength(200.0f), showGrid(true), 
          showTrajectories(false), showConflicts(true), showTrails(true), // Trails ON by default
          showHUD(true), showDroneLabels(true), cameraAngleX(45.0f), 
          cameraAngleY(-45.0f), cameraDistance(1000.0f),
          cameraTarget(0, 0, 150), frameCount(0), fpsTimer(0), currentFPS(0),
          zoomVelocity(0), rotVelocityX(0), rotVelocityY(0),
          currentScenario(1), timeScale(1.0f), 
          currentTrajectoryFile("No file loaded"), currentAnalysisFile("") {}
    
    void initialize() {
        displayCache.initialize();
    }
    
    void updateCameraInertia(float deltaTime) {
        zoomVelocity *= 0.9f;
        rotVelocityX *= 0.8f;
        rotVelocityY *= 0.8f;
        
        if (fabs(zoomVelocity) > 0.1f) {
            cameraDistance += zoomVelocity * deltaTime * 60.0f;
            if (cameraDistance < 100.0f) cameraDistance = 100.0f;
            if (cameraDistance > 2000.0f) cameraDistance = 2000.0f;
        }
        
        if (fabs(rotVelocityX) > 0.1f || fabs(rotVelocityY) > 0.1f) {
            cameraAngleX += rotVelocityY * deltaTime * 60.0f;
            cameraAngleY += rotVelocityX * deltaTime * 60.0f;
            
            if (cameraAngleX > 89.0f) cameraAngleX = 89.0f;
            if (cameraAngleX < -89.0f) cameraAngleX = -89.0f;
            while (cameraAngleY > 360.0f) cameraAngleY -= 360.0f;
            while (cameraAngleY < 0.0f) cameraAngleY += 360.0f;
        }
    }
    
    void render(const std::vector<std::unique_ptr<Drone>>& drones, 
                const JSONCollisionLoader& collisionLoader, 
                float simTime, float deltaTime) {
        // Update FPS
        frameCount++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0f) {
            currentFPS = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0;
        }
        
        // Update camera inertia
        updateCameraInertia(deltaTime);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // 3D Scene
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0f, 1280.0f/720.0f, 1.0f, 5000.0f);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        // Calculate camera position
        float radX = cameraAngleX * M_PI / 180.0f;
        float radY = cameraAngleY * M_PI / 180.0f;
        Vector3D cameraPos(
            cameraTarget.x + cameraDistance * cos(radX) * cos(radY),
            cameraTarget.y + cameraDistance * cos(radX) * sin(radY),
            cameraTarget.z + cameraDistance * sin(radX)
        );
        
        gluLookAt(cameraPos.x, cameraPos.y, cameraPos.z,
                  cameraTarget.x, cameraTarget.y, cameraTarget.z,
                  0, 0, 1);
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POINT_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        
        // Draw scene
        if (showGrid) drawGrid();
        drawAxes();
        
        // Draw trails
        if (showTrails) {
            for (const auto& drone : drones) {
                if (drone->getIsActive()) {
                    drawTrail(*drone, cameraPos);
                }
            }
        }
        
        // Draw trajectories
        if (showTrajectories) {
            for (const auto& drone : drones) {
                drawTrajectory(*drone, cameraPos);
            }
        }
        
        // Draw collision zones
        if (showConflicts) {
            auto collisions = collisionLoader.getAllCollisions();
            for (const auto& collision : collisions) {
                drawCollisionZone(collision, simTime);
            }
        }
        
        // Draw drones
        glEnable(GL_LIGHTING);
        
        // Update light position to follow camera
        GLfloat lightPos0[] = {cameraPos.x, cameraPos.y, cameraPos.z + 500.0f, 1.0f};
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
        
        // Draw drones (draw primary drone last so it's on top)
        std::vector<const Drone*> regularDrones;
        const Drone* primaryDrone = nullptr;
        
        for (const auto& drone : drones) {
            if (drone->getIsActive()) {
                if (drone->getIsPrimary()) {
                    primaryDrone = drone.get();
                } else {
                    regularDrones.push_back(drone.get());
                }
            }
        }
        
        // Draw regular drones first
        for (const auto& drone : regularDrones) {
            drawDrone(*drone, cameraPos);
        }
        
        // Draw primary drone last (so it's always visible)
        if (primaryDrone) {
            drawDrone(*primaryDrone, cameraPos);
        }
        
        // Draw HUD
        if (showHUD) {
            drawHUD(drones, collisionLoader, simTime);
        }
        
        glutSwapBuffers();
    }
    
    void rotateCamera(float dx, float dy) {
        rotVelocityX += dx * 0.05f;
        rotVelocityY += dy * 0.05f;
    }
    
    void zoomCamera(float delta) {
        zoomVelocity += delta * 0.5f;
    }
    
    void panCamera(float dx, float dy) {
        float scale = cameraDistance * 0.002f;
        cameraTarget.x += dx * scale;
        cameraTarget.y -= dy * scale;
    }
    
    void toggleGrid() { showGrid = !showGrid; }
    void toggleTrajectories() { showTrajectories = !showTrajectories; }
    void toggleConflicts() { showConflicts = !showConflicts; }
    void toggleTrails() { showTrails = !showTrails; }
    void toggleHUD() { showHUD = !showHUD; }
    void toggleLabels() { showDroneLabels = !showDroneLabels; }
    
    void resetCamera() {
        cameraAngleX = 45.0f;
        cameraAngleY = -45.0f;
        cameraDistance = 500.0f;
        cameraTarget = Vector3D(0, 0, 50);
        zoomVelocity = 0;
        rotVelocityX = 0;
        rotVelocityY = 0;
    }
    
    // Setters for HUD info
    void setTimeScale(float scale) { timeScale = scale; }
    void setTrajectoryFile(const std::string& filename) { 
        currentTrajectoryFile = filename; 
    }
    void setAnalysisFile(const std::string& filename) { 
        currentAnalysisFile = filename; 
    }
};

// ============================================================================
// JSON Trajectory Loader
// ============================================================================
class JSONTrajectoryLoader {
private:
    std::mt19937 rng;
    
    Vector3D randomColor() {
        std::uniform_real_distribution<float> dist(0.3f, 1.0f);
        return Vector3D(dist(rng), dist(rng), dist(rng));
    }
    
public:
    JSONTrajectoryLoader() {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    bool loadFromJSON(const std::string& filename, 
                     std::vector<std::unique_ptr<Drone>>& drones,
                     int maxDrones = Config::MAX_DRONES,
                     bool loadPrimary = false) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open JSON file: " << filename << std::endl;
            return false;
        }
        
        try {
            json root;
            file >> root;
            
            if (root.is_object() && root.contains("id") && 
                root["id"].get<std::string>() == "primary_drone") {
                // Load primary drone
                if (!loadPrimary) return false;
                
                std::cout << "Loading primary drone from: " << filename << std::endl;
                
                auto drone = std::make_unique<Drone>(0, Vector3D(0.0f, 1.0f, 0.0f), true);
                
                Trajectory traj;
                json waypoints = root["waypoints"];
                
                for (const auto& wp : waypoints) {
                    float time = wp["time"].get<float>();
                    float x = wp["x"].get<float>();
                    float y = wp["y"].get<float>();
                    float z = wp["z"].get<float>();
                    
                    traj.addWaypoint(Waypoint(Vector3D(x, y, z), time, 1.0f));
                }
                
                drone->setTrajectory(traj, true);
                drones.insert(drones.begin(), std::move(drone));
                std::cout << "Primary drone loaded with " << waypoints.size() << " waypoints" << std::endl;
                return true;
            } else {
                // Load regular drones
                if (loadPrimary) return false;
                
                json metadata = root["metadata"];
                int totalDrones = metadata["num_drones"].get<int>();
                float totalTime = metadata["total_time"].get<float>();
                float timeStep = metadata["time_step"].get<float>();
                
                std::cout << "Loading " << totalDrones << " drones from JSON..." << std::endl;
                std::cout << "Total time: " << totalTime << "s, Time step: " << timeStep << "s" << std::endl;
                
                int dronesToLoad = std::min(totalDrones, maxDrones);
                
                json jsonDrones = root["drones"];
                int loadedCount = 0;
                
                for (int i = 0; i < dronesToLoad && i < jsonDrones.size(); i++) {
                    json droneData = jsonDrones[i];
                    std::string droneId = droneData["id"].get<std::string>();
                    
                    int droneNum = 0;
                    if (droneId.find("drone_") != std::string::npos) {
                        droneNum = std::stoi(droneId.substr(6));
                    }
                    
                    auto drone = std::make_unique<Drone>(droneNum, randomColor(), false);
                    
                    Trajectory traj;
                    json waypoints = droneData["waypoints"];
                    
                    int wpCount = waypoints.size();
                    int targetWaypoints = 200;
                    int sampleStep = std::max(1, wpCount / targetWaypoints);
                    
                    for (int w = 0; w < wpCount; w += sampleStep) {
                        const auto& wp = waypoints[w];
                        float time = wp["time"].get<float>();
                        float x = wp["x"].get<float>();
                        float y = wp["y"].get<float>();
                        float z = wp["z"].get<float>();
                        
                        float arrivalRadius = timeStep * sampleStep * 5.0f;
                        traj.addWaypoint(Waypoint(Vector3D(x, y, z), time, arrivalRadius));
                    }
                    
                    if (!waypoints.empty()) {
                        const auto& lastWp = waypoints[wpCount - 1];
                        float time = lastWp["time"].get<float>();
                        float x = lastWp["x"].get<float>();
                        float y = lastWp["y"].get<float>();
                        float z = lastWp["z"].get<float>();
                        float arrivalRadius = timeStep * 5.0f;
                        
                        if ((wpCount - 1) % sampleStep != 0) {
                            traj.addWaypoint(Waypoint(Vector3D(x, y, z), time, arrivalRadius));
                        }
                    }
                    
                    drone->setTrajectory(traj, true);
                    drones.push_back(std::move(drone));
                    loadedCount++;
                    
                    if (loadedCount % 10 == 0) {
                        std::cout << "Loaded " << loadedCount << " drones..." << std::endl;
                    }
                }
                
                std::cout << "Successfully loaded " << loadedCount << " regular drones from JSON." << std::endl;
                return true;
            }
        }
        catch (const json::exception& e) {
            std::cerr << "Error: JSON parsing error: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
    }
};

// ============================================================================
// Enhanced Simulation Controller with JSON Collision Loading
// ============================================================================
class Simulation {
private:
    std::vector<std::unique_ptr<Drone>> drones;
    JSONCollisionLoader collisionLoader;
    OpenGLRenderer renderer;
    JSONTrajectoryLoader trajectoryLoader;
    float simulationTime;
    float timeScale;
    bool isPaused;
    int droneCount;
    float lastFrameTime;
    std::string currentTrajectoryFile;
    std::string currentAnalysisFile;
    
    // For random colors
    std::mt19937 rng;
    
    Vector3D randomColor() {
        std::uniform_real_distribution<float> dist(0.3f, 1.0f);
        return Vector3D(dist(rng), dist(rng), dist(rng));
    }
    
public:
    Simulation(const std::string& droneFile = "", 
               const std::string& primaryFile = "",
               const std::string& analysisFile = "") 
        : simulationTime(0), timeScale(1.0f), isPaused(false), 
          droneCount(0), lastFrameTime(0), 
          currentTrajectoryFile(droneFile),
          currentAnalysisFile(analysisFile) {
        
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        renderer.initialize();
        renderer.setTimeScale(timeScale);
        
        // Load primary drone first
        if (!primaryFile.empty()) {
            if (trajectoryLoader.loadFromJSON(primaryFile, drones, 1, true)) {
                std::cout << "✓ Primary drone loaded from: " << primaryFile << std::endl;
                renderer.setTrajectoryFile("Primary: " + primaryFile);
            } else {
                std::cout << "✗ Failed to load primary drone from: " << primaryFile << std::endl;
            }
        }
        
        // Load regular drones
        if (!droneFile.empty()) {
            if (trajectoryLoader.loadFromJSON(droneFile, drones, Config::MAX_DRONES, false)) {
                droneCount = drones.size();
                if (!primaryFile.empty()) droneCount--;
                renderer.setTrajectoryFile("Drones: " + droneFile);
                std::cout << "✓ Regular drones loaded from: " << droneFile << std::endl;
            } else {
                std::cout << "✗ Failed to load regular drones from: " << droneFile << std::endl;
            }
        }
        
        // Load collision analysis
        if (!analysisFile.empty()) {
            if (collisionLoader.loadFromJSON(analysisFile)) {
                renderer.setAnalysisFile(analysisFile);
                std::cout << "✓ Collision analysis loaded from: " << analysisFile << std::endl;
            } else {
                std::cout << "✗ Failed to load collision analysis from: " << analysisFile << std::endl;
            }
        }
        
        std::cout << "\nTotal drones in simulation: " << drones.size() << std::endl;
        std::cout << "Primary drone: " << (drones.empty() ? "No" : (drones[0]->getIsPrimary() ? "Yes" : "No")) << std::endl;
    }
    
    bool loadDroneTrajectory(const std::string& filename) {
        currentTrajectoryFile = filename;
        
        // Remove all non-primary drones
        auto it = std::remove_if(drones.begin(), drones.end(), 
            [](const std::unique_ptr<Drone>& d) { return !d->getIsPrimary(); });
        drones.erase(it, drones.end());
        
        if (trajectoryLoader.loadFromJSON(filename, drones, Config::MAX_DRONES, false)) {
            droneCount = drones.size();
            for (const auto& drone : drones) {
                if (drone->getIsPrimary()) {
                    droneCount--;
                    break;
                }
            }
            renderer.setTrajectoryFile("Drones: " + filename);
            std::cout << "Loaded drone trajectories from: " << filename << std::endl;
            return true;
        }
        return false;
    }
    
    bool loadPrimaryTrajectory(const std::string& filename) {
        auto it = std::remove_if(drones.begin(), drones.end(), 
            [](const std::unique_ptr<Drone>& d) { return d->getIsPrimary(); });
        drones.erase(it, drones.end());
        
        if (trajectoryLoader.loadFromJSON(filename, drones, 1, true)) {
            renderer.setTrajectoryFile("Primary: " + filename);
            std::cout << "Loaded primary drone from: " << filename << std::endl;
            return true;
        }
        return false;
    }
    
    bool loadCollisionAnalysis(const std::string& filename) {
        currentAnalysisFile = filename;
        if (collisionLoader.loadFromJSON(filename)) {
            renderer.setAnalysisFile(filename);
            std::cout << "Loaded collision analysis from: " << filename << std::endl;
            return true;
        }
        return false;
    }
    
    void update(float deltaTime) {
        if (isPaused) return;
        
        float scaledDelta = deltaTime * timeScale;
        simulationTime += scaledDelta;
        
        float totalDuration = 200.0f;
        
        if (simulationTime >= totalDuration) {
            simulationTime = 0.0f;
            
            for (auto& drone : drones) {
                drone->reset();
            }
            
            std::cout << "🔄 Simulation restarted at 0s (200s reached)" << std::endl;
        }
        
        // Update all drones
        for (auto& drone : drones) {
            drone->update(scaledDelta);
        }
        
        // Update drone colors based on collisions
        auto currentCollisions = collisionLoader.getCollisionsAtTime(simulationTime, 1.0f);
        
        // Reset all drone colors first
        for (auto& drone : drones) {
            drone->resetColor();
        }
        
        // Color drones involved in current collisions
        for (const auto& collision : currentCollisions) {
            int droneNum = 0;
            if (collision.conflicting_drone_id.find("drone_") != std::string::npos) {
                try {
                    droneNum = std::stoi(collision.conflicting_drone_id.substr(6));
                } catch (...) {
                    continue;
                }
            }
            
            for (auto& drone : drones) {
                if (drone->getId() == droneNum && !drone->getIsPrimary()) {
                    if (collision.severity == "HIGH") {
                        drone->setColor(Vector3D(1, 0.2f, 0));
                    } else if (collision.severity == "MEDIUM") {
                        drone->setColor(Vector3D(1, 0.6f, 0));
                    } else {
                        drone->setColor(Vector3D(1, 0.9f, 0));
                    }
                    break;
                }
            }
        }
    }
    
    void render(float deltaTime) {
        renderer.render(drones, collisionLoader, simulationTime, deltaTime);
    }
    
    void reset() {
        simulationTime = 0;
        for (auto& drone : drones) {
            drone->reset();
        }
    }
    
    void togglePause() { isPaused = !isPaused; }
    void setTimeScale(float scale) { 
        timeScale = scale; 
        renderer.setTimeScale(scale);
    }
    void rotateCamera(float dx, float dy) { renderer.rotateCamera(dx, dy); }
    void zoomCamera(float delta) { renderer.zoomCamera(delta); }
    void panCamera(float dx, float dy) { renderer.panCamera(dx, dy); }
    void resetCamera() { renderer.resetCamera(); }
    void toggleGrid() { renderer.toggleGrid(); }
    void toggleTrajectories() { renderer.toggleTrajectories(); }
    void toggleConflicts() { renderer.toggleConflicts(); }
    void toggleTrails() { renderer.toggleTrails(); }
    void toggleHUD() { renderer.toggleHUD(); }
    void toggleLabels() { renderer.toggleLabels(); }
    
    float getSimulationTime() const { return simulationTime; }
    bool getIsPaused() const { return isPaused; }
    int getDroneCount() const { return droneCount; }
    
    void printStatus() {
        auto collisions = collisionLoader.getAllCollisions();
        auto currentCollisions = collisionLoader.getCollisionsAtTime(simulationTime, 1.0f);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "SIMULATION STATUS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(2) << simulationTime << "s" << std::endl;
        std::cout << "Total Drones: " << drones.size() << std::endl;
        
        int primaryCount = 0;
        for (const auto& drone : drones) {
            if (drone->getIsPrimary()) primaryCount++;
        }
        std::cout << "  Primary: " << primaryCount << std::endl;
        std::cout << "  Regular: " << (drones.size() - primaryCount) << std::endl;
        
        std::cout << "Collision Analysis: " << currentAnalysisFile << std::endl;
        std::cout << "Total Collisions in Analysis: " << collisions.size() << std::endl;
        std::cout << "Active Collisions (now ±1s): " << currentCollisions.size() << std::endl;
        
        for (const auto& drone : drones) {
            if (drone->getIsPrimary()) {
                auto pos = drone->getPosition();
                std::cout << "\nPrimary Drone:" << std::endl;
                std::cout << "  Position: (" << std::fixed << std::setprecision(1) 
                         << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
                std::cout << "  Speed: " << std::setprecision(1) << drone->getSpeed() << " m/s" << std::endl;
                break;
            }
        }
        
        if (!currentCollisions.empty()) {
            std::cout << "\nCurrent Collisions:" << std::endl;
            for (const auto& collision : currentCollisions) {
                std::cout << "  " << collision.primary_drone_id << " <-> " 
                         << collision.conflicting_drone_id 
                         << " at t=" << std::fixed << std::setprecision(1) << collision.time << "s" 
                         << ", distance=" << std::setprecision(2) << collision.distance << "m"
                         << ", severity=" << collision.severity << std::endl;
            }
        }
        
        std::map<std::string, int> severityCount;
        for (const auto& c : collisions) {
            severityCount[c.severity]++;
        }
        
        if (!severityCount.empty()) {
            std::cout << "\nCollisions by Severity:" << std::endl;
            for (const auto& [severity, count] : severityCount) {
                std::cout << "  " << severity << ": " << count << std::endl;
            }
        }
        
        std::cout << "========================================\n" << std::endl;
    }
};

// ============================================================================
// Global Variables and GLUT Callbacks
// ============================================================================
Simulation* simulation = nullptr;
int lastMouseX = 0, lastMouseY = 0;
bool mouseLeftDown = false;
bool mouseRightDown = false;
bool mouseMiddleDown = false;
auto lastFrameTime = std::chrono::high_resolution_clock::now();

void display() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
    lastFrameTime = currentTime;
    
    simulation->render(deltaTime);
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glutPostRedisplay();
}

void timer(int value) {
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
    lastFrameTime = currentTime;
    
    simulation->update(deltaTime);
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:
            exit(0);
            break;
        case ' ':
            simulation->togglePause();
            std::cout << (simulation->getIsPaused() ? "⏸  Paused" : "▶  Running") << std::endl;
            break;
        case 'r':
        case 'R':
            simulation->reset();
            std::cout << "↻ Reset" << std::endl;
            break;
        case 'n':
        case 'N':
            {
                std::string filename;
                std::cout << "Enter JSON drone trajectory file path: ";
                std::cin.ignore();
                std::getline(std::cin, filename);
                if (!simulation->loadDroneTrajectory(filename)) {
                    std::cout << "Failed to load drone trajectory file." << std::endl;
                }
            }
            break;
        case 'p':
        case 'P':
            {
                std::string filename;
                std::cout << "Enter JSON primary drone trajectory file path: ";
                std::cin.ignore();
                std::getline(std::cin, filename);
                if (!simulation->loadPrimaryTrajectory(filename)) {
                    std::cout << "Failed to load primary drone trajectory file." << std::endl;
                }
            }
            break;
        case 'a':
        case 'A':
            {
                std::string filename;
                std::cout << "Enter JSON collision analysis file path: ";
                std::cin.ignore();
                std::getline(std::cin, filename);
                if (!simulation->loadCollisionAnalysis(filename)) {
                    std::cout << "Failed to load collision analysis file." << std::endl;
                }
            }
            break;
        case 'g':
        case 'G':
            simulation->toggleGrid();
            break;
        case 't':
        case 'T':
            simulation->toggleTrails();
            break;
        case 'c':
        case 'C':
            simulation->toggleConflicts();
            break;
        case 'l':
        case 'L':
            simulation->toggleTrajectories();
            break;
        case 'h':
        case 'H':
            simulation->toggleHUD();
            break;
        case 'b':
        case 'B':
            simulation->toggleLabels();
            break;
        case 'v':
        case 'V':
            simulation->resetCamera();
            std::cout << "📷 Camera reset" << std::endl;
            break;
        case '1':
            simulation->setTimeScale(0.5f);
            std::cout << "Speed: 0.5x" << std::endl;
            break;
        case '2':
            simulation->setTimeScale(1.0f);
            std::cout << "Speed: 1.0x" << std::endl;
            break;
        case '3':
            simulation->setTimeScale(2.0f);
            std::cout << "Speed: 2.0x" << std::endl;
            break;
        case '4':
            simulation->setTimeScale(5.0f);
            std::cout << "Speed: 5.0x" << std::endl;
            break;
        case 's':
        case 'S':
            simulation->printStatus();
            break;
    }
    glutPostRedisplay();
}

void specialKeys(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_UP:
            simulation->zoomCamera(-20.0f);
            break;
        case GLUT_KEY_DOWN:
            simulation->zoomCamera(20.0f);
            break;
        case GLUT_KEY_LEFT:
            simulation->rotateCamera(-5.0f, 0);
            break;
        case GLUT_KEY_RIGHT:
            simulation->rotateCamera(5.0f, 0);
            break;
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    lastMouseX = x;
    lastMouseY = y;
    
    if (button == GLUT_LEFT_BUTTON) {
        mouseLeftDown = (state == GLUT_DOWN);
    } else if (button == GLUT_RIGHT_BUTTON) {
        mouseRightDown = (state == GLUT_DOWN);
    } else if (button == GLUT_MIDDLE_BUTTON) {
        mouseMiddleDown = (state == GLUT_DOWN);
    } else if (button == 3) {
        simulation->zoomCamera(-20.0f);
    } else if (button == 4) {
        simulation->zoomCamera(20.0f);
    } else if (state == GLUT_UP) {
        mouseLeftDown = mouseRightDown = mouseMiddleDown = false;
    }
    
    glutPostRedisplay();
}

void motion(int x, int y) {
    float dx = x - lastMouseX;
    float dy = y - lastMouseY;
    
    if (mouseLeftDown) {
        simulation->rotateCamera(dx * 0.5f, -dy * 0.5f);
    } else if (mouseRightDown) {
        simulation->zoomCamera(dy * 1.0f);
    } else if (mouseMiddleDown) {
        simulation->panCamera(dx, dy);
    }
    
    lastMouseX = x;
    lastMouseY = y;
    glutPostRedisplay();
}

void printInstructions() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  UAV DECONFLICTION SYSTEM - ENHANCED PRIMARY DRONE        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\n📋 ENHANCED PRIMARY DRONE VISIBILITY:" << std::endl;
    std::cout << "  • 40x larger than regular drones" << std::endl;
    std::cout << "  • Bright green pulsing glow" << std::endl;
    std::cout << "  • Long glowing breadcrumb trail" << std::endl;
    std::cout << "  • Searchlight beam pointing down" << std::endl;
    std::cout << "  • Compass showing direction" << std::endl;
    std::cout << "  • Velocity vector arrow" << std::endl;
    std::cout << "  • Enhanced HUD panel" << std::endl;
    std::cout << "\n🎮 CONTROLS:" << std::endl;
    std::cout << "  SPACE          Pause/Resume" << std::endl;
    std::cout << "  R              Reset simulation" << std::endl;
    std::cout << "  V              Reset camera" << std::endl;
    std::cout << "  G              Toggle grid" << std::endl;
    std::cout << "  T              Toggle trails" << std::endl;
    std::cout << "  C              Toggle collision zones" << std::endl;
    std::cout << "  L              Toggle trajectory lines" << std::endl;
    std::cout << "  H              Toggle HUD" << std::endl;
    std::cout << "  B              Toggle labels" << std::endl;
    std::cout << "  1-4            Set speed (0.5x to 5x)" << std::endl;
    std::cout << "  Mouse          Rotate/Zoom/Pan" << std::endl;
    std::cout << "\n⚠ LOOK FOR THE LARGE GREEN PULSING SPHERE!" << std::endl;
    std::cout << "   It leaves a glowing trail showing its path." << std::endl;
    std::cout << "\n═══════════════════════════════════════════════════════════════\n" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char** argv) {
    // Hardcoded file paths
    std::string droneFile = "/home/arka/Trajectra/waypoint_generation/drone_waypoints.json";
    std::string primaryFile = "/home/arka/Trajectra/waypoint_generation/primary_waypoint.json";
    std::string analysisFile = "/home/arka/Trajectra/trajectory_control/detailed_analysis.json";
    
    std::cout << "Starting UAV Deconfliction Simulation with Enhanced Primary Drone..." << std::endl;
    std::cout << "Loading files:" << std::endl;
    std::cout << "  1. Primary drone: " << primaryFile << std::endl;
    std::cout << "  2. Regular drones: " << droneFile << std::endl;
    std::cout << "  3. Collision analysis: " << analysisFile << std::endl;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(1280, 720);
    glutInitWindowPosition(50, 50);
    glutCreateWindow("UAV Deconfliction System - ENHANCED PRIMARY DRONE VISUALIZER");
    
    // OpenGL initialization
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    
    // Enhanced lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    
    GLfloat lightPos0[] = { 500.0f, 500.0f, 1000.0f, 1.0f };
    GLfloat lightPos1[] = { -500.0f, -500.0f, 500.0f, 1.0f };
    GLfloat lightAmbient[] = { 0.3f, 0.3f, 0.35f, 1.0f };
    GLfloat lightDiffuse[] = { 0.8f, 0.8f, 0.9f, 1.0f };
    GLfloat lightSpecular[] = { 0.5f, 0.5f, 0.6f, 1.0f };
    
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);
    
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightDiffuse);
    
    // Create simulation with all three files
    simulation = new Simulation(droneFile, primaryFile, analysisFile);
    
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(16, timer, 0);
    
    printInstructions();
    
    if (simulation->getDroneCount() == 0) {
        std::cout << "\n⚠ WARNING: No drones loaded!" << std::endl;
        std::cout << "The JSON files may not exist or may be invalid." << std::endl;
        std::cout << "Press 'N' to load drone trajectories manually." << std::endl;
        std::cout << "Press 'P' to load primary drone manually." << std::endl;
        std::cout << "Press 'A' to load collision analysis manually." << std::endl;
    } else {
        std::cout << "\n✅ Successfully loaded simulation!" << std::endl;
        std::cout << "   Total drones: " << simulation->getDroneCount() << std::endl;
    }
    
    std::cout << "\n📊 Simulation starting..." << std::endl;
    std::cout << "   🔍 LOOK FOR THE LARGE GREEN PULSING PRIMARY DRONE!" << std::endl;
    std::cout << "   🚁 It has a glowing trail showing its exact path." << std::endl;
    
    lastFrameTime = std::chrono::high_resolution_clock::now();
    
    glutMainLoop();
    
    delete simulation;
    return 0;
}