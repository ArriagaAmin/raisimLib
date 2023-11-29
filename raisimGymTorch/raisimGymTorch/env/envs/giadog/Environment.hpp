#pragma once

#include <map>
#include <set>
#include <cmath>
#include <chrono>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "utils.hpp"
#include "EnvConfig.hpp"
#include "RaisimGymEnv.hpp"
#include "ContactSolver.hpp"
#include "ExternalForce.hpp"
#include "HeightScanner.hpp"
#include "WorldGenerator.hpp"

#define GOAL_RADIUS 0.2

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
 * CONSTANT The robot must not spin, just follow a constant cartesain command.
 *
 */
enum command_t
{
    STRAIGHT,
    STANCE,
    STATIC_SPIN,
    RANDOM,
    EXTERNAL,
    PROBABILITY,
    FIXED_DIRECTION
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
        command_t current_command_mode_ = command_t::STRAIGHT; // Preventive initialization
        // Probability of chosing a command mode:
        // We have 4 cases:
        // Case 1: Move straight facing the target.
        // Case 2: Move straight facing a random direction.
        // Case 3: Spin in place.
        // Case 4: Stand still.
        // Case 5: Move in a fixed direction. (No spin, and the cartesian command is constant)
        double case_1_prob_, case_2_prob_, case_3_prob_, case_4_prob_, case_5_prob_;
        // Command duration
        double command_duration_ = 3.0; // 3 seconds

        // Indicates if the robot may have to spinning while moving towards
        // its target.
        bool spinning_;
        // Indicates if the robot may have to face another direction while
        // moving towards its target.
        bool change_facing_;
        // Turning direction: 1 for clockwise, -1 for counter-clockwise
        // and to not rotate 0.
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
        
        // Vector parallel to the x component of the rotation matrix of the robot.
        Eigen::Vector3d x_component_vector_;
        // Vector parallel to the y component of the rotation matrix of the robot.
        Eigen::Vector3d y_component_vector_;
        // Vector parallel to the z component of the rotation matrix of the robot.
        Eigen::Vector3d z_component_vector_;

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

        // Minimun desired speed of the robot to consider it is fluently on the terrain
        double traversability_min_speed_treshold_ = 0.066; // We set a default value of 0.066 m/s

        // Terminal state
        bool terminal_state_ = false;

        // Ability of the robot to transit the current terrain
        double traversability_;
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
        
        Eigen::VectorXd regular_observations_;
        Eigen::VectorXd privileged_observations_;
        Eigen::VectorXd historic_observations_;
        Eigen::VectorXd observations_vector_;

        std::vector<std::string> historic_observations_keys_;
        std::vector<std::string> regular_observations_keys_;
        std::vector<std::string> privileged_observations_keys_;

        std::map<std::string, int> observations_sizes_;

        // We must create a list with the default order of the observations
        std::vector<std::string> defualt_obs_order = {
                "target_direction",
                "turning_direction",
                "body_height",
                "gravity_vector",
                "linear_velocity",
                "angular_velocity",
                "joint_position",
                "joint_velocity",
                "FTG_sin_phases",
                "FTG_cos_phases",
                "FTG_frequencies",
                "base_frequency",
                "joint_pos_err_hist",
                "joint_vel_hist",
                "feet_target_hist",
                "terrain_normal",
                "feet_height_scan",
                "foot_contact_forces",
                "foot_contact_states",
                "shank_contact_states",
                "thigh_contact_states",
                "foot_ground_fricction",
                "external_force",
                "pd_constants"
        };

       
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
            {"traversability", 0.0},
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
        // Indicates which visual objects will be displayed
        bool display_target_, display_direction_, display_turning_,
            display_height_, display_x_component_, display_y_component_, 
            display_z_component_, display_linear_vel_, display_angular_vel_;
        // Visual objects that will display environment data
        raisim::Visuals *visual_target_, *direction_head_, *turning_head_,
            *x_component_head_, *y_component_head_, *z_component_head_,
            *linear_vel_head_, *angular_vel_head_;
        // Visual objects that will display environment data
        raisim::PolyLine *direction_body_, *turning_body_, *height_line_,
            *x_component_body_, *y_component_body_, *z_component_body_,
            *linear_vel_body_, *angular_vel_body_;
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
                std::chrono::steady_clock::now().time_since_epoch().count())};

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
        void change_target(bool preserve_command_mode = false);

        /**
         * @brief Updates the target direction, and target direction angle.
         * Calculates the height of the terrain at the new target position.
         * Places the visual target at that position and that specific
         * height.
         *
         * @param preserve_command If true, the current 
         */
        void update_target(void);

        /**
         * @brief Updates the position of the visual objects.
         *
         */
        void update_visual_objects(void);

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
        // We also need the size of the observations
        int regular_obs_begin_idx_;
        int privileged_obs_begin_idx_;
        int historic_obs_begin_idx_;
        
        int regular_obs_size_;
        int privileged_obs_size_;
        int historic_obs_size_;
        int obs_size_;
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
            std::vector<std::string> non_privileged_obs,
            std::vector<std::string> privileged_obs,
            std::vector<std::string> historic_obs,
            int port);

        /**
         * @brief Reset simulation and start new episode.
         *
         * @param epoch Current train epoch
         * @return step_t Current environment information
         */
        step_t reset() final;

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
         * @param amplitude Height of the hills.
         * @param roughness Terrain roughness.
         */
        void hills(double frequency, double amplitude, double roughness);

        /**
         * @brief Create the training terrain that contains stairs.
         *
         * @param width Width of each step.
         * @param height Height of each step.
         */
        void stairs(double width, double height);

        //** Fast stairs **//
        /**
         * @brief Create the training terrain that contains stairs.
         * This method is faster than the other one.
         * 
        */
        void fast_stairs(const std::vector<double> &heigth_map,
                    double total_length,
                    double total_width,
                    int resolution);

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


        /**
         * @brief Updates the secondary reward coefficient
         * 
        */
        void update_curriculum_coefficient(void);

        /**
         * @brief Changes the secondary reward coefficient
         * 
         * @param value New secondary reward coefficient
         * 
        */
        void set_curriculum_coefficient(double value);
        
        /**
         * @brief Returns a dictionary with the indexes of the observations
         *        in the observation vector
        */
        std::map<std::string, std::array<int, 2>> get_observations_indexes(void);


        /**
         * @brief Allows to place the robot in a specific position. Parameters 
         * with NaN values will be ignored and the current value will be placed
         *
         * @param x Absolute x position
         * @param y Absolute y position
         * @param z Absolute z position
         * @param roll Roll angle position
         * @param pitch Pitch angle position
         * @param yaw Yaw angle position
         *
         */
        void set_absolute_position(
            double x,
            double y,
            double z,
            double roll,
            double pitch,
            double yaw
            );

        /**
         * @brief Allows to set an absolute speed to the robot. Parameters 
         * with NaN values will be ignored and the current value will be placed
         *
         * @param linear_x Absolute x linear velocity
         * @param linear_y Absolute y linear velocity
         * @param linear_z Absolute z linear velocity
         * @param angular_x Absolute x angular velocity
         * @param angular_y Absolute y angular velocity
         * @param angular_z Absolute z angular velocity
         *
         */
        void set_absolute_velocity(
            double linear_x,
            double linear_y,
            double linear_z,
            double angular_x,
            double angular_y,
            double angular_z);
        
        /**
         * @brief Allows to set the target foot positions of the robot and the 
         * base position. This function is used to have the robot float in space and 
         * to test the Inverse Kinematics algorithm
         * 
         * @param foot_pos Foot positions (a 12 element vector) represents the foot 
         * positions in cartesian space
         * The order is LF_xyz, RF_xyz, LB_xyz RB_xyz
         * 
         * @param x Absolute x position
         * @param y Absolute y position
         * @param z Absolute z position
         * @param roll Roll angle position
         * @param pitch Pitch angle position
         * @param yaw Yaw angle position
         * 
         * 
        */
       void set_foot_positions_and_base_pose(
        const Eigen::Ref<EigenVec> &foot_pos,
        double x,
        double y,
        double z,
        double roll,
        double pitch,
        double yaw
        );

        /**
         * @brief Allows to set the gait configuration parameters "on the fly"
         * This is to tune the quadruped gait interactively.
         * 
         * @param foot_pos
         * 
         * 
        */
       void set_gait_config(
            double base_frequency,
            double leg_1_phase,
            double leg_2_phase,
            double leg_3_phase,
            double leg_4_phase,
            double foot_vertical_span,
            double angular_movement_delta,
            double x_movement_delta,
            double y_movement_delta,
            double leg_span,
            bool use_horizontal_frame
        );
    };
}