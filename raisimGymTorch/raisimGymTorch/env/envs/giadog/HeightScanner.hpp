#pragma once

#include <stack>
#include "EnvConfig.hpp"
#include "RaisimGymEnv.hpp"

/**
 * @brief Class that calculates the height of terrain around the robot feet.
 *
 */
class HeightScanner
{
private:
    // Number of scans for each ring.
    int scans_per_ring_;
    // Number of rings.
    int n_scan_rings_;
    // Number of scans per foot.
    int scans_per_foot_;
    // Number of legs of the robot.
    int n_legs_;
    // Innermost ring radius.
    double foot_scan_radius_;

    // Indicates if the scans will be displayed.
    bool visualizable_;
    // Environment where the simulation occurs.
    raisim::World *world_;
    // Robot that is in the simulation.
    raisim::ArticulatedSystem *anymal_;
    // Id of the robot feet in the simulation.
    std::vector<int> feet_frames_idx_;
    // Name of the objects that represent the feet of the robot in the
    // simulation.
    std::vector<std::string> feet_link_names_;
    // List of pointers to the objects that will visually represent the
    // scans in the simulation.
    std::vector<raisim::Visuals *> foot_height_scan_visuals_;

    /**
     * @brief Get the points to do the height scan around the foot
     *
     * @param x   Foot x coordinat in world frame
     * @param y   Foot x coordinat in world frame
     * @param yaw Robot base yaw in world frame
     *
     * @return Points to do the height scan around the foot
     */
    std::vector<Eigen::Vector2d> footScanCoordinates_(
        double x,
        double y,
        double yaw);

public:
    // Total number of scans.
    int n_scans_;
    // Scans of each foot.
    Eigen::VectorXd feet_height_scan;
    // Position of each foot.
    Eigen::VectorXd current_feet_position;
    // Square of the speed of the legs. This speed is used in reward function.
    Eigen::Vector4d feet_speed_squared_;
    // Reward for keeping the leg above the ground when flexed.
    double foot_clearance_reward;

    HeightScanner(void){};

    /**
     * @param world Environment where the simulation occurs.
     * @param anymal Robot that is in the simulation.
     * @param env_config Simulation environment configuration parameters.
     * @param feet_link_names List of the names of the links of the robot
     *       that are the feet (These can be found in the  urdf file of the robot).
     * @param visualizable Wether or not to visualize the height scan.
     *
     */
    HeightScanner(
        raisim::World *world,
        raisim::ArticulatedSystem *anymal,
        const EnvConfig *env_config,
        std::vector<std::string> feet_link_names,
        bool visualizable = false);

    /**
     * @brief Perform the height scan around the robot foot.
     *
     * @param base_yaw The yaw angle of the robot base.
     */
    void foot_scan(double base_yaw);

    /**
     * @brief Calculate the foot clearance reward given the feet position.
     */
    double clearance_reward(const Eigen::Vector4d &foot_phases);

    /**
     * @brief Add spheres to the simulation to visualize the scan.
     *
     * @param server A pointer to the raisim server where the visuals shapes
     *      will be added
     */
    void add_visual_indicators(raisim::RaisimServer *server);
};
