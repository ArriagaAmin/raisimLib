#pragma once

#include "RaisimGymEnv.hpp"

/**
 * @brief Class that is used to apply an external force to the robot base,
 * and to get the external force expresed in the robot base frame.
 */
class ExternalForceApplier
{
private:
    // Robot that is in the simulation.
    raisim::ArticulatedSystem *anymal_;
    // Time during which the external force will be applied to the robot.
    double time_threshold_;
    // Maximum magnitude of the external force applied to the robot.
    double maximun_magnitude_;

public:
    // External force applied to the robot.
    Eigen::Vector3d external_force_world_frame, external_force_base_frame;

    ExternalForceApplier(void){};

    /**
     * @param anymal Robot that is in the simulation.
     * @param time_step_threshold Time during which the external force will
     *      be applied to the robot.
     * @param external_force_maximun_magnitude Maximum posible magnitude
     *      of the external force.
     */
    ExternalForceApplier(
        raisim::ArticulatedSystem *anymal,
        double time_step_threshold,
        double external_force_maximun_magnitude);

    /**
     * @brief Applies the external force to the robot base
     *
     * @param time_step Current time step in the environment.
     *
     */
    void apply_external_force(double time_step);

    /**
     * @brief Gets the external force applied to the robot base in the base
     * frame
     *
     * @param R_base_to_world The rotation matrix from the base frame to
     *      the world frame.
     *
     */
    void external_force_in_base(Eigen::Matrix3d R_base_to_world);
};
