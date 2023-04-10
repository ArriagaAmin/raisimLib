#pragma once

#include <map>
#include <set>
#include <cmath>
#include <chrono>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#include "utils.hpp"
#include "EnvConfig.hpp"
#include "RaisimGymEnv.hpp"
#include "ContactSolver.hpp"
#include "ExternalForce.hpp"
#include "HeightScanner.hpp"
#include "WorldGenerator.hpp"

#define GOAL_RADIUS 0.2
#define MIN_DESIRED_VEL 0.066

/**
 * @brief Possible types of terrain.
 *
 */
enum terrain_t
{
    HILLS,
    STAIRS,
    CELULAR_STEPS,
    STEPS,
    SLOPE,
    DEMO
};

/**
 * @brief Different ways in which the robot applies a command.
 * STRAIGHT means that the robot must go to a point that changes when it reaches it.
 * STANCE is that the robot must not move.
 * STATIC_SPIN indicates that the robot must turn without moving.
 * RANDOM means a random command among the above.
 * EXTERNAL is that the robot is controlled by an external user.
 *
 */
enum command_t
{
    STRAIGHT,
    STANCE,
    STATIC_SPIN,
    RANDOM,
    EXTERNAL
};

// Possible directions the robot may be facing while moving.
const std::vector<double>
    POSIBLE_FACING_ANGLES = {
        0.0,
        M_PI_4,
        -M_PI_4,
        M_PI_2,
        -M_PI_2,
        3 * M_PI_4,
        -3 * M_PI_4,
        M_PI,
        -M_PI,
        0.0};

namespace raisim
{
    class ENVIRONMENT : public RaisimGymEnv
    {
    private:
        // Terrain generator
        WorldGenerator generator_;
        // The robot's body.
        raisim::ArticulatedSystem *anymal_;
        // Environment configuration.
        EnvConfig env_config_;

        // Number of robot joints.
        int n_joints_;
        // Robot position target
        Eigen::VectorXd pos_target_;
        // Robot velocity target
        Eigen::VectorXd vel_target_;

        // Dimension of generalized coordinate
        int generalized_coord_dim_;
        // Generalized coordinate of the robot
        // Configuracion: [x, y, z, qr, qi, qj, qk, 12 joints angles]
        Eigen::VectorXd generalized_coord_;
        // Initial values for generalized coordinate of the robot
        Eigen::VectorXd generalized_coord_init_;

        // Dimension of generalized velocity
        int generalized_vel_dim_;
        // Generalized velocity of the robot
        Eigen::VectorXd generalized_vel_;
        // Initial values for generalized velocity of the robot
        Eigen::VectorXd generalized_vel_init_;

        // Joint PD (Proportional plus Derivative) controller constants.
        Eigen::VectorXd pd_constants_;
        // Joint PD (Proportional plus Derivative) controller noise.
        double p_max_, p_min_, d_max_, d_min_;

        // Dimenson of actions
        int action_dim_;
        // The action to apply to the robot.
        Eigen::VectorXd action_;
        // Mean actions taken.
        Eigen::VectorXd action_mean_;
        // Vector to scale the received action to the actual action ranges.
        Eigen::VectorXd action_scale_;

        // Real command mode
        command_t command_mode_;
        // Current command mode
        command_t current_command_mode_;
        // Indicates if the robot may have to spinning while moving towards
        // its target.
        bool spinning_;
        // Indicates if the robot may have to face another direction while
        // moving towards its target.
        bool change_facing_;
        // Turning direction: 1 for clockwise, -1 for counter-clockwise
        // and to not rotate.
        int turning_direction_ = 0;
        // Angle the robot should be looking at while moving.
        double facing_angle_ = 0.0;
        // Angle of the robot with respect to the command direction.
        double target_angle_;
        // Target position within the environment.
        Eigen::Vector2d target_position_;
        // Target direction of the robot.
        Eigen::Vector2d target_direction_;
        // Foot Trajectory Generator frequencies.
        Eigen::Vector4d FTG_frequencies_;
        // Sine of the Foot Trajectory Generator phases.
        Eigen::Vector4d FTG_sin_phases_;
        // Cosine of the Foot Trajectory Generator phases.
        Eigen::Vector4d FTG_cos_phases_;
        // Foot Trajectory Generator phases.
        Eigen::Vector4d FTG_phases_;
        // Objective angles for the joints.
        Eigen::VectorXd joint_target_;
        // Target position for feet.
        Eigen::VectorXd feet_target_pos_;
        // History of joint objective positions.
        Eigen::VectorXd feet_target_hist_;
        // Robot orientation.
        Eigen::Vector3d base_euler_;
        // Joint positions.
        Eigen::VectorXd joint_position_;
        // Joint velocities.
        Eigen::VectorXd joint_velocity_;
        // Joint accelerations.
        Eigen::VectorXd joint_acceleration_;
        // Gravity vector. It is parallel to the z component of the
        // rotation matrix of the robot.
        Eigen::Vector3d gravity_vector_;
        // History of errors between the objective and the observed
        // velocity of the joints.
        Eigen::VectorXd joint_vel_hist_;
        // History of errors in position of the joints.
        Eigen::VectorXd joint_pos_err_hist_;
        // Robot linear velocity.
        Eigen::Vector3d linear_vel_;
        // Robot angular velocity.
        Eigen::Vector3d angular_vel_;
        // Height scanner
        HeightScanner height_scanner_;
        // Contact info solver.
        ContactSolver contact_solver_;
        // External force applier to the robot base.
        ExternalForceApplier external_force_applier_;
        // Ability of the robot to transit the current terrain
        double traverability_;
        // Height of the robot body;
        double body_height_;
        // Orientation noise standard deviation.
        double orientation_noise_std_;

        // Indicates if noise should be enabled in observations.
        bool noise_;
        // Curriculum coefficient
        double curriculum_coeff_;
        // Curriculum base
        double curriculum_base_;
        // Curriculum decay
        double curriculum_decay_;
        // Set of observations of the environment
        std::map<std::string, Eigen::VectorXd> observations_ = {
            // Target direction of the robot.
            {"target_direction", Eigen::VectorXd::Zero(2)},
            // Turning direction: 1 for clockwise, -1 for counter-clockwise
            // and to not rotate.
            {"turning_direction", Eigen::VectorXd::Zero(1)},
            // Robot body height.
            {"body_height", Eigen::VectorXd::Zero(1)},
            // Gravity vector
            {"gravity_vector", Eigen::VectorXd::Zero(3)},
            // Robot linear velocity
            {"linear_velocity", Eigen::VectorXd::Zero(3)},
            // Robot angular velocity
            {"angular_velocity", Eigen::VectorXd::Zero(3)},
            // All joint angles
            {"joint_position", Eigen::VectorXd::Zero(12)},
            // All joint velocities
            {"joint_velocity", Eigen::VectorXd::Zero(12)},
            // Foot Trajectory Generator sine phases
            {"FTG_sin_phases", Eigen::VectorXd::Zero(4)},
            // Foot Trajectory Generator cosine phases
            {"FTG_cos_phases", Eigen::VectorXd::Zero(4)},
            // Foot Trajectory Generator frequency
            {"FTG_frequencies", Eigen::VectorXd::Zero(4)},
            // Robot base frequency
            {"base_frequency", Eigen::VectorXd::Zero(1)},
            // Historical of errors between the indicated position and the one
            // that was actually obtained
            {"joint_pos_err_hist", Eigen::VectorXd::Zero(24)},
            // Historical of joint velocities
            {"joint_vel_hist", Eigen::VectorXd::Zero(24)},
            // Historical of feet position targets
            {"feet_target_hist", Eigen::VectorXd::Zero(24)},
            // Terrain normal vector at each foot
            {"terrain_normal", Eigen::VectorXd::Zero(12)},
            // Terrain height scan at each foot. This have dynamic size
            {"feet_height_scan", Eigen::VectorXd::Zero(1)},
            // Contact force at each foot
            {"foot_contact_forces", Eigen::VectorXd::Zero(4)},
            // Contact state at each foot
            {"foot_contact_states", Eigen::VectorXd::Zero(4)},
            // Contact state at each shank
            {"shank_contact_states", Eigen::VectorXd::Zero(4)},
            // Contact state at each thigh
            {"thigh_contact_states", Eigen::VectorXd::Zero(4)},
            // Ground friction at each foot
            {"foot_ground_fricction", Eigen::VectorXd::Zero(4)},
            // External force applied to base
            {"external_force", Eigen::VectorXd::Zero(3)},
            // PD constants
            {"pd_constants", Eigen::VectorXd::Zero(24)}};
        // Set of observations of the environment
        std::map<std::string, double> observations_noise_ = {
            {"linear_velocity", 0.07},
            {"angular_velocity", 0.2},
            {"joint_position", 0.314},
            {"joint_velocity", 0.0314}};
        // Other relevant information about the state of the environment
        // that is not part of the observations
        std::map<std::string, double> info_ = {
            // Ability of the robot to move through the terrain
            {"traverability", 0.0},
            // Dimensionless number that classifies the robot's walk
            {"froude", 0.0},
            // Projected speed on movement command
            {"projected_speed", 0.0},
            // Estimated maximum torque of the robot
            {"max_torque", 0.0},
            // Estimated power used by the robot
            {"power", 0.0}};

        // Current terrain type
        terrain_t terrain_;
        // Current epoch
        int epoch_ = 0;
        // Episode duration
        double episode_duration_;
        // Episode elapsed time.
        double elapsed_time_;
        // Episode elapsed timesteps.
        int elapsed_steps_;
        // Indicates if latency variations should be added.
        bool variable_latency_;
        // If variable latency is enabled, this will be the peak of the
        // distribution, if not, it is the constant simulation latency value
        double latency_;
        // Triangular distribution for Latency control.
        std::piecewise_linear_distribution<double> latency_distribution_;

        // Port through which you can connect to the simulation visualizer.
        int port_;
        // Indicates if the scans will be displayed.
        bool visualizable_ = false;
        // Visual representation of the target position.
        raisim::Visuals *visual_target_;
        // ID of the robot feet in the simulation
        std::set<size_t> foot_indexes_;

        // Normal distribution.
        std::normal_distribution<double> norm_dist_;
        // Uniform distribution ranging from minus one to one.
        std::uniform_int_distribution<> minus_one_one_dist{-1, 1};
        // Uniform distribution ranging from zero to one.
        std::uniform_real_distribution<> zero_one_real_dist_{0, 1};
        // Pseudo-random generator of 32-bit numbers.
        std::mt19937 merssen_twister_{
            static_cast<unsigned int>(
            std::chrono::steady_clock::now().time_since_epoch().count()
            )
        };
        
        // Pseudo-random generator of 32-bit numbers.
        thread_local static std::mt19937 random_gen_;

        /**
         * @brief Get the terrain height in a point
         *
         * @param x Point position X
         * @param y Point position Y
         * @return double Terrain height in the point
         */
        double get_terrain_height(double x, double y);

        /**
         * @brief Calculate the PD from the current robot state
         */
        void set_pd_gains(void);

        /**
         * @brief Change the target position (goal), and the turning
         * direction. Recalculates the turning direction as a random integer
         * between -1 and 1. Recalculates the  target direction angle and
         * sets the new target position, at 1.5 meters away from the current
         * position.
         *
         */
        void change_target(void);

        /**
         * @brief Updates the target direction, and target direction angle.
         * Calculates the height of the terrain at the new target position.
         * Places the visual target at that position and that specific
         * height.
         *
         */
        void update_target(void);

        /**
         * @brief Updates the position of the visual target when command_mode
         * is STRAIGHT. Calculates the height of the terrain at the new target
         * position. Places the visual target at that position and that specific
         * height.
         *
         */
        void update_visual_target(void);

        /**
         * @brief Updates the environment observation.
         * In this environment, the observation is composed of the following:
         * - The target direction [2]
         * - The turning direction [1]
         * - body height [1]
         * - body orientation (given by the gravity vector) [3]
         * - body linear velocity [3]
         * - body angular velocity [3]
         * - body joint angles [nJoints] (12)
         * - body joint velocities [nJoints] (12)
         * - Trajectory generator state [24] (4 frecuencies, 4 cosine
         *      phases, 4 sine phases, 12 foot target positions)
         */
        void update_observations(void);

        /**
         * @brief Update other relevant information about the state of the
         * environment that is not part of the observations
         *
         */
        void update_info(void);

        /**
         * @brief Updates the current bouncing reward based on the state of
         * the environment
         *
         */
        void register_rewards(void);

        /**
         * @brief Check if the current state of the robot is terminal,
         * that is, if the robot fell or reached its goal
         *
         * @return Indicates if the current state of the robot is terminal
         */
        bool is_terminal_state(void);

    public:
        /**
         * @param resource_dir Directory where the resources needed to build
         *      the environment are located
         * @param cfg Environment configuration file
         * @param visualizable Indicates if the robot target will be
         *      displayed.
         * @param port
         *
         */
        ENVIRONMENT(
            const std::string &resource_dir,
            const Yaml::Node &cfg,
            bool visualizable,
            int port);

        /**
         * @brief Reset simulation and start new episode.
         *
         * @param epoch Current train epoch
         * @return step_t Current environment information
         */
        step_t reset(int epoch) final;

        /**
         * @brief Steps the simulation.
         *
         * @param action Action taken by the robot.
         * @return step_t Environment information after applying the action
         */
        step_t step(const Eigen::Ref<EigenVec> &action) final;

        /**
         * @brief Create the training terrain that contains hills.
         *
         * @param frequency How often each hill appears.
         *      Recommended range: [0.00, 0.35]
         * @param amplitude Height of the hills.
         *      Recommended range: [0.00, 2.00]
         * @param roughness Terrain roughness.
         *      Recommended range: [0.00, 0.06]
         */
        void hills(double frequency, double amplitude, double roughness);

        /**
         * @brief Create the training terrain that contains stairs.
         *
         * @param width Width of each step.
         * @param height Height of each step.
         */
        void stairs(double width, double height);

        /**
         * @brief Create the training terrain that contains stepped terrain
         *
         * @param frequency Frequency of the cellular noise
         * @param amplitude Scale to multiply the cellular noise
         */
        void cellular_steps(double frequency, double amplitude);

        /**
         * @brief Generates a terrain made of steps (little square boxes)
         *
         * @param width  Width of each of the steps [m]
         * @param height Amplitude of the steps[m]
         */
        void steps(double width, double height);

        /**
         * @brief Generates a terrain made of a slope.
         *
         * @param slope  The slope of the slope [m]
         */
        void slope(double slope, double roughness);

        /**
         * @brief Sets the robot command direction. This method is used when
         * the robot command type is external.
         *
         * @param target_angle Angle to which the robot must move
         * @param turning_direction Turning direction: 1 for clockwise, -1
         * for counter-clockwise and to not rotate.
         * @param stop The robot must not move.
         */
        void set_command(double target_angle, int turning_direction, bool stop) final;

        /**
         * @brief Gets the dimensions of all observations
         * 
         * @return std::map<std::string, int> Map each dictionary to its 
         * respective dimension
         */
        std::map<std::string, int> get_observations_dimension(void);

        /**
         * @brief Get action space dimension
         * 
         * @return int Action space dimension
         */
        int get_action_dimension(void);
    };

}