#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "RaisimGymEnv.hpp"

/**
 * @brief Class thats is used to get the contcat states of the robot leg
 * articulations, the friction coeficients of the legs and the forces and
 * normal vectors to the robot feet. The contact states are stored in the
 * attributes of the class. It also calculates the undesirable_collisions_reward_
 * which is used to penalize the collision of the robot shanks and thighs
 * with the ground.
 */
class ContactSolver
{
private:
    // Differential time.
    double dt_;
    // Environment where the simulation occurs.
    raisim::World *world_;
    // Robot that is in the simulation.
    raisim::ArticulatedSystem *quadruped_;
    // Id of objects that is in the simulation.
    std::vector<int> shank_ids_, thigh_ids_, foot_ids_, foot_links_ids_;

public:
    // Direction of the normal force on each leg of the robot.
    Eigen::VectorXd terrain_normal;
    // Status of contacts on parts of the robot.
    Eigen::Vector4d thigh_contact_states,
        shank_contact_states,
        foot_contact_states;
    // Contact force of robot legs.
    Eigen::Vector4d foot_contact_forces;
    // Friction force on the robot's legs.
    Eigen::Vector4d foot_ground_friction;
    // Reward obtained for undesirable collisions.
    double undesirable_collisions_reward_;

    ContactSolver(void){};

    /**
     * @brief Construct a new Contact And Fricction Info Solver object.
     *
     * @param world Pointer to the Environment.
     * @param quadruped Pointer to the Robot that is in the Environment.
     * @param simulation_dt The simulation time step (Timestep of the
     *      raisim simulation, not the control loop time step).
     * @param fricction_coeff_mean Mean of the fricction coefficient
     *      between the foot and the terrain.
     * @param fricction_coeff_std  Standard deviation of the fricction
     *      coefficient between the foot and the terrain.
     * @param thigh_parent_names List of the names of the thigh parent articulations (i.e the hips)
     *      objects of the robot.
     * @param shank_parent_names List of the names of the shank parent articulations (i.e the shanks)
     *      objects of the robot.
     * @param foot_parent_names  List of the names of the foot parent articulations (i.e the shanks)
     *      objects of the robot.
     * @param foot_link_names    List of the names of the foot links (i.e the feet)
     *
     * @details The thigh_names, shank_names and foot_names are the names
     * of the frames of the robot. There is a distinction in raisim between
     * the links, the frames and the joints. The order of the links is:
     *  * Front left,
     *  * Front right,
     *  * Back left,
     *  * Back right
     */
    ContactSolver(
        raisim::World *world,
        raisim::ArticulatedSystem *quadruped,
        double simulation_dt,
        double fricction_coeff_mean,
        double fricction_coeff_std,
        std::vector<std::string> thigh_parent_names,
        std::vector<std::string> shank_parent_names,
        std::vector<std::string> foot_parent_names,
        std::vector<std::string> foot_link_names
        );

    /**
     * @brief Computes the contact states and contact forces for the legs.
     * This function is called every control time step.
     */
    void contact_info(void);
};
