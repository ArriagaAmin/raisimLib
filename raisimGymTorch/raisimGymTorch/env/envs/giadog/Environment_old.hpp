#pragma once

#include <map>
#include <set>
#include <cmath>
#include <chrono> 
#include <random>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#include "EnvConfig.hpp" 
#include "ContactInfo.hpp" 
#include "RaisimGymEnv.hpp" 
#include "ExternalForce.hpp" 
#include "HeightScanner.hpp" 
#include "WorldGenerator.hpp" 
#include "ControlPipeline.hpp" 
#include "Quaternion2Euler.hpp" 

namespace raisim 
{
    class ENVIRONMENT : public RaisimGymEnv 
    {
        private:
            // The robot's body.
            raisim::ArticulatedSystem* anymal_;
            // Robot body height.
            double body_height_;
            // Dimension of generalized coordinate of the robot.
            int gc_dim_;
            // Dimension of generalized velocity of the robot.
            int gv_dim_;
            // Number of robot joints.
            int n_joints_;
            // ID of the robot feet in the simulation
            std::set<size_t> foot_indexes_;
            // Initial generalized coordinate of the robot.
            Eigen::VectorXd gc_init_;
            // Initial generalized velocity of the robot.
            Eigen::VectorXd gv_init_;
            // Generalized coordinate of the robot.
            Eigen::VectorXd gc_;
            // Generalized velocity of the robot.
            Eigen::VectorXd gv_;

            // Episode elapsed time.
            double episode_elapsed_time_;
            // Episode elapsed timesteps.
            int episode_elapsed_steps_;
            // Duracion del episodio.
            double episode_duration_;
            // Environment configuration.
            EnvConfig env_config_;
            // Contact info solver.
            ContactSolver contact_solver_;
            // Height scanner.
            HeightScanner height_scanner_;
            // External force applier.
            ExternalForceApplier external_force_applier_;

            // Terrain generator.
            WorldGenerator generator;
            // Possible types of terrain.
            enum TERRAIN_TYPES_ {HILLS, STAIRS, CELULAR_STEPS, STEPS, SLOPE};
            // Current type of terrain.
            TERRAIN_TYPES_ terrain_type_;
            // Terrain dimensions in meters.
            double terrain_x_size_, terrain_y_size_; 

            // Port through which you can connect to the simulation visualizer.
            int port_;
            // Indicates if the scans will be displayed.
            bool visualizable_ = false;

            // The action to apply to the robot.
            Eigen::VectorXd action_scaled_;
            // Mean actions taken.
            Eigen::VectorXd action_mean_;
            // Vector to scale the received action to the actual action ranges.
            Eigen::VectorXd actionScale_;

            // Joint PD (Proportional plus Derivative) controller constants.
            Eigen::VectorXd pd_constants_;
            // Joint PD (Proportional plus Derivative) controller noise.
            double p_max_, p_min_, d_max_, d_min_;

            // Turning direction: 1 for clockwise, -1 for counter-clockwise
            // and to not rotate.
            int turning_direction_;
            // Angle of the robot with respect to the command direction.
            double target_direction_angle_;
            // Target direction of the robot.
            Eigen::Vector2d target_direction_;
            // Target position within the environment.
            Eigen::Vector2d target_position_;
            // Robot position target.
            Eigen::VectorXd p_target_;
            // Robot velocity target.
            Eigen::VectorXd v_target_;
            // Robot linear velocity.
            Eigen::Vector3d body_linear_vel_;
            // Robot angular velocity.
            Eigen::Vector3d body_angular_vel_; 
            // Robot orientation.
            Eigen::Vector3d base_euler_; 
            // Gravity vector. It is parallel to the z component of the 
            // rotation matrix of the robot.
            Eigen::Vector3d body_rotation_matrix_z_component_;
            // Objective angles for the joints.
            Eigen::VectorXd joints_target_;
            // Target position for feet.
            Eigen::VectorXd feet_target_pos_;
            // Foot Trajectory Generator frequencies.
            Eigen::Vector4d FTG_frequencies_;
            // Sine of the Foot Trajectory Generator phases.
            Eigen::Vector4d FTG_sin_phases_ ;
            // Cosine of the Foot Trajectory Generator phases.
            Eigen::Vector4d FTG_cos_phases_;
            // Foot Trajectory Generator phases.
            Eigen::Vector4d FTG_phases_;
            // History of errors between the objective and the observed
            // position of the joints.
            Eigen::VectorXd joint_pos_err_hist_;
            // History of errors between the objective and the observed
            // velocity of the joints.
            Eigen::VectorXd joint_vel_hist_;
            // History of joint objective positions.
            Eigen::VectorXd feet_target_hist_;
            // Joint positions.
            Eigen::VectorXd joint_position_;
            // Joint velocities.
            Eigen::VectorXd joint_velocity_;
            // Joint acceleration.
            Eigen::VectorXd joint_acceleration_;
            // Environment Observations Vector.
            Eigen::VectorXd ob_double_;

            // Map that goes from the observations to the indices they occupy
            // in the observations vector
            std::map<std::string, std::vector<int>> observations_idx_map_;
            // Indicates if noise should be enabled in observations.
            bool enable_noise_;
            // Dictionary that maps the variable names to the noise standar 
            // deviations
            std::map<std::string, double> noise_std_map_;
            // Vector to scale the noise to the actual ranges.
            Eigen::VectorXd noise_scaler_vec_;
            // Orientation noise standard deviation.
            double orientation_noise_std_;
            
            // Reward function variables
            double proj_angular_vel_, proj_linear_vel_, w_2_, ort_vel_; 
            double torque_reward_, linear_vel_reward_, angular_vel_reward_;
            double base_motion_reward_, traverability_;
            Eigen::Vector2d h_angular_vel_, h_linear_vel_;
            // reward curriculm coefficient
            double curriculum_coeff_, curriculum_coeff_decay_;

            // Indicates if the gait will be directed. If so, the robot should 
            // always face the target. Otherwise, the robot may have to face 
            // another direction while moving towards its target.
            bool directed_gait_;
            // Different ways in which the robot applies a command. STRAIGHT 
            // means that the robot must go to a point that changes when it 
            // reaches it. FIXED is that the robot command does not change.
            // And STANCE is that the robot must not move.
            enum COMMAND_MODE_ {STRAIGHT, FIXED, STANCE};
            // Current mode in which the robot applies a command.
            COMMAND_MODE_ current_command_mode_;
            // Possible angles the robot should be looking at while moving.
            std::vector<double> posible_facing_angles_ = {
                0.0     , M_PI_4    , -M_PI_4, M_PI_2, -M_PI_2, 
                3*M_PI_4, -3*M_PI_4 , M_PI   , -M_PI , 0.0
            };
            // Probabilities that the robot will obtain the different types 
            // of commands
            double straight_prob_, 
                straight_with_random_direction_prob_,
                fixed_prob_,
                stance_prob_;
            double command_duration_;
            // Angle the robot should be looking at while moving.
            double facing_angle_ = 0.0;
            // Indicates that the robot is controlled by an external user.
            bool external_command_;

            // Uniform distribution ranging from minus one to one.
            std::uniform_int_distribution<> minus_one_one_dist{-1, 1};
            // Uniform distribution ranging from zero to one.
            std::uniform_real_distribution<> zero_one_real_dist_{0, 1};
            // Distribucion normal.
            std::normal_distribution<double> norm_dist_;
            // Pseudo-random generator of 32-bit numbers.
            std::mt19937 merssen_twister_{ static_cast<unsigned int>(
                std::chrono::steady_clock::now().time_since_epoch().count()
            ) };
            // Pseudo-random generator of 32-bit numbers.
            thread_local static std::mt19937 random_gen_;
            // Indicates if latency variations should be added.
            bool variable_latency_enabled_;
            // Triangular distribution for Latency control.
            std::piecewise_linear_distribution<double> latency_distribution_;

            // Visual representation of the target position.
            raisim::Visuals *visual_target_;
            // Visual representation of the turning direction of the robot.
            raisim::Visuals *turning_arrow_head_; 
            // Visual representation of the turning direction of the robot.
            raisim::PolyLine *turning_arrow_body_;
            // Visual representation of the direction of the robot.
            raisim::PolyLine *direction_arrow_body_; 
            // Visual representation of the direction of the robot.
            raisim::Visuals *direction_arrow_head_;

            /**
             * @brief Defines the observation space of the environment.
             * 
             * @param enabled_observations : List of enabled observations.
             * 
             * DELETE
             * 
             */
            void defineObservationSpace_(Yaml::Node observation_params);

            /**
             * @brief Defines the current command 
             * 
             * DELETE
             * 
             */
            void setCommandMode_();

            /**
             * @brief Create the training terrain that contains hills.
             * 
             * @param frequency How often each hill appears.
             * @param amplitude Height of the hills.
             * @param roughness Terrain roughness.
             * 
             */
            void hills_(double frequency, double amplitude, double roughness);

            /**
             * @brief Create the training terrain that contains stairs.
             * 
             * @param width Width of each step.
             * @param height Height of each step.
             */
            void stairs_(double width, double height);

            /**
             * @brief Create the training terrain that contains stepped terrain 
             * 
             * @param frequency Frequency of the cellular noise
             * @param amplitude Scale to multiply the cellular noise 
            */
            void cellularSteps_(double frequency, double amplitude);

            /**
             * @brief Generates a terrain made of steps (little square boxes)
             * 
             * @param width  Width of each of the steps [m]
             * @param height Amplitude of the steps[m]
             */
            void steps_(double width, double height);

            /**
             * @brief Generates a terrain made of a slope.
             * 
             * @param slope  The slope of the slope [m]
             */
            void slope_(double slope, double roughness);

            /**
             * @brief Updates the position of the visual target.
             * Calculates the height of the terrain at the new target position.
             * Places the visual target at that position and that specific 
             * height.
             * 
             */
            void updateVisualTarget_(void);

            /**
             * @brief Updates the target direction, and target direction angle.
             * Calculates the height of the terrain at the new target position.
             * Places the visual target at that position and that specific 
             * height.
             * 
             */
            void updateTargetDirection_(void);

            /**
             * @brief Updates the target position (goal), and the turning 
             * direction. Recalculates the turning direction as a random integer
             * between -1 and 1. Recalculates the  target direction angle and 
             * sets the new target position, at 1.5 meters away from the current
             * position.
             * 
             */
            void updateTargetPositionAndTurningDirection_(void);


             /**
             * @brief Updates the visual shapes that indicate the command of the
             * robot.
             * 
             */
            void updateVisualTargetDirection_(void);

            /**
             * @brief Get the Base Euler Angles object
             * 
             * @param ea 
             */
            void updateBaseEulerAngles_(Eigen::Ref<EigenVec> ea);

            /**
             * @brief Updates the environment observation.
             * Specifically the ob_double_ variable.
             * In this environment, the observation is composed of the following:
             * - The target direction (2D normalized vector)
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
            void updateObservation_(void);

            /**
             * @brief Registers the rewards.
             * 
             */
            void registerRewards_(void);

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
            explicit ENVIRONMENT(
                const std::string& resource_dir, 
                const Yaml::Node& cfg, 
                bool visualizable,
                int port
            );
            
            /**
             * @brief Dummy function.
             * 
             */
            void init(void) final { }

            /**
             * @brief Resets the simulation to the initial state.
             * 
             */
            void reset(void) final;

            /**
             * @brief Updates the robot's observations vector.
             * 
             * @param ob Vector to contain the observations.
             * 
             */
            void observe(Eigen::Ref<EigenVec> ob) final;

            /**
             * @brief Perform a time step within the simulation.
             * 
             * @param action Action taken by the robot.
             * 
             * @return Reward obtained in this time step.
             * 
             */
            float step(const Eigen::Ref<EigenVec>& action) final;

            /**
             * @brief Check if the current state of the robot is terminal, 
             * that is, if the robot fell or reached its goal
             * 
             * @param terminalReward 
             * 
             * @return Indicates if the current state of the robot is terminal
             */
            bool isTerminalState(void) final;

            /**
             * @brief Get the Traversability object
             * 
             * @param trav 
             */
            double getTraversability(void);

            /**
             * @brief Returns the power used by the robot.
             * 
             * 
             */
            double getPower(void);

            /**
             * @brief Returns the froude number of the robot.
             * 
             */
            double getFroude(void);

            /**
             * @brief Returns the robot orthogonal speed [m/s]
             * 
             */
            double getProjSpeed(void);

            /**
             * @brief Returns the maximun torque applied by the motors of the robot. [N,m]
             * 
             */
            double getMaxTorque(void);

            /**
             * @brief Sets the robot command direction.
             * 
             */
            void setCommand(
                double direction_angle, 
                double turning_direction, 
                bool stop
            );

            /**
             * @brief Sets the robot PD gains. Depending on the configuration 
             * file it can be a default value or a random value from a uniform 
             * distribution. Check the configuration file for more information 
             * (cfg.yaml).
             * 
             */
            void setPDGains_(void);

    };
    thread_local std::mt19937 raisim::ENVIRONMENT::random_gen_;

    ENVIRONMENT::ENVIRONMENT(
        const std::string& resource_dir, 
        const Yaml::Node& cfg, 
        bool visualizable,
        int port
    ) : 
        RaisimGymEnv(resource_dir, cfg, port), 
        visualizable_(visualizable), 
        norm_dist_(0, 1),
        port_(port)
    {
        // Initialize the environment configuration.
        Yaml::Node gait = cfg["gait"], 
                   robot = cfg["robot"],
                   control_dt = cfg["control_dt"], 
                   control = cfg["control"],
                   reward_control = cfg["reward_control"],
                   action_params = cfg["action_params"],
                   height_scan = cfg["height_scan"],
                   latency_control_params = cfg["latency_control"],
                   pd_gains = cfg["pd_gains"];
            
        this->env_config_.SIGMA_0[0]     = gait["leg_1_phase"].template As<double>();
        this->env_config_.SIGMA_0[1]     = gait["leg_2_phase"].template As<double>();
        this->env_config_.SIGMA_0[2]     = gait["leg_3_phase"].template As<double>();
        this->env_config_.SIGMA_0[3]     = gait["leg_4_phase"].template As<double>();
        this->env_config_.BASE_FREQUENCY = gait["base_frequency"].template As<double>();
        this->env_config_.H              = gait["max_foot_height"].template As<double>();

        this->orientation_noise_std_ = cfg["orientation_noise_std"].template As<double>();

        this->env_config_.CARTESIAN_DELTA = gait["cartesian_movement_heuristic"].template As<bool>();
        this->env_config_.ANGULAR_DELTA   = gait["angular_movement_heuristic"].template As<bool>();

        this->env_config_.H_OFF     = robot["H_OFF"].template As<double>();
        this->env_config_.V_OFF     = robot["V_OFF"].template As<double>();
        this->env_config_.THIGH_LEN = robot["THIGH_LEN"].template As<double>();
        this->env_config_.SHANK_LEN = robot["SHANK_LEN"].template As<double>();
        this->env_config_.LEG_SPAN  = robot["LEG_SPAN"].template As<double>();
        this->env_config_.CONTROL_DT = control_dt.template As<double>();
        this->env_config_.VEL_TH = reward_control["velocity_threshold"].template As<double>();

        this->env_config_.EXTERNAL_FORCA_MAX_VALUE     = cfg["external_force_max_value"].template As<double>();
        this->env_config_.EXTERNAL_FORCA_TIME_THRESHOLD = cfg["external_force_time_threshold"].template As<double>();
        
        this->directed_gait_ = control["directed_gait"].template As<bool>();
        
        this->straight_prob_   = control["move_straight_command_prob"].template As<double>();
        this->straight_with_random_direction_prob_ = \
            control["move_straight_facing_random_direction_command_prob"].template As<double>();
        this->fixed_prob_ = control["fixed_direction_command_prob"].template As<double>();
        this->stance_prob_          = control["stance_command_prob"].template As<double>();
        this->command_duration_             = control["command_duration"].template As<double>();

        

        // Reward curriculum initizaliation
        this->curriculum_coeff_ = reward_control["reward_curriculum_coeff"].template As<double>();
        this->curriculum_coeff_decay_ = reward_control["reward_curriculum_coeff_decay"].template As<double>();
        

        this->env_config_.N_SCAN_RINGS     = height_scan["n_scan_rings"].template As<int>();
        this->env_config_.SCANS_PER_RING   = height_scan["scans_per_ring"].template As<int>();
        this->env_config_.FOOT_SCAN_RADIUS = height_scan["foot_scan_radius"].template As<double>();

        // Create world
        this->world_ = std::make_unique<raisim::World>();

        this->episode_duration_ = cfg["max_time"].template As<double>();

        // PD gains
        this->p_max_ = pd_gains["kp_max"].template As<double>();
        this->p_min_ = pd_gains["kp_min"].template As<double>();
        this->d_max_ = pd_gains["kd_max"].template As<double>();
        this->d_min_ = pd_gains["kd_min"].template As<double>();
       

        // Add robot
        this->anymal_ = this->world_->addArticulatedSystem(
            this->resource_dir_ + "/giadog/mini_ros/urdf/spot.urdf",
            "",
            {},
            raisim::COLLISION(2),// Collision group 
            -1
        );
        this->anymal_->setName("GiaDoG");
        this->anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        // Terrain generator
        this->generator = WorldGenerator(this->world_.get(), this->anymal_);
        this->generator.hills(0.0, 0.0, 0.0);

        // Get robot data
        this->gc_dim_   = this->anymal_->getGeneralizedCoordinateDim();
        this->gv_dim_   = this->anymal_->getDOF();
        this->n_joints_ = this->gv_dim_ - 6;

        // Initialize containers
        this->gc_.setZero(this->gc_dim_); 
        this->gv_.setZero(this->gv_dim_); 
        this->gc_init_.setZero(this->gc_dim_);
        this->gv_init_.setZero(this->gv_dim_);
        this->p_target_.setZero(this->gc_dim_); 
        this->v_target_.setZero(this->gv_dim_);

        // We perform a raycast to get the height of the ground around 
        // the x = 0, y = 0. This is used to set the initial height of the 
        // robot. This is important because the robot is not standing on 
        // the ground when it is created.
        auto& col = this->world_->rayTest({0,0,10}, {0,0,-1}, 50., true);
        double z_init = 0.25;
        if (col.size() > 0)
        {
            z_init += col[0].getPosition()[2];
        }

        // This is nominal configuration of anymal
        // Configuracion: [x, y, z, qr, qi, qj, qk, 12 joints angles]
        this->gc_init_ << 0, 0, z_init, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, 
            -0.8, -0.03, 0.4, -0.8, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8;
        // Set pd gains
        this->pd_constants_.setZero(this->n_joints_ * 2);
        setPDGains_();

        this->anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(this->gv_dim_));
        
        this->actionDim_ = 16; 
        // Setting vectors to zero
        this->action_mean_.setZero(this->actionDim_); 
        this->actionScale_.setZero(this->actionDim_);
        
        Yaml::Node action_scale = action_params["action_scale"];
        Yaml::Node action_mean = action_params["action_mean"];
        for (int i = 0; i < this->actionDim_ - 4; i++) 
        {
            // if the mod 3 of i is zero
            if (i % 3 == 0)
            {
                this->actionScale_[i] = action_scale["x_residual"].template As<double>();
                this->action_mean_[i] = action_mean["x_residual"].template As<double>();
            }
           else if (i%3 == 1)
            {
                this->actionScale_[i] = action_scale["y_residual"].template As<double>();
                this->action_mean_[i] = action_mean["y_residual"].template As<double>();
            }
            else if (i%3 == 2)
            {
                this->actionScale_[i] = action_scale["z_residual"].template As<double>();
                this->action_mean_[i] = action_mean["z_residual"].template As<double>();
            }            
        }
        for (int i = this->actionDim_ - 4; i < this->actionDim_; i++) 
        {
            this->actionScale_[i] = action_scale["frequency"].template As<double>(); 
            this->action_mean_[i] = action_mean["frequency"].template As<double>(); 
        }
        // EULER ANGLES (Orientation of the robot)
        this->base_euler_.setZero(3);
        // JOINT TARGETS
        this->joints_target_.setZero(12);  
        // FTG DATA
        this->feet_target_pos_.setZero(12);
        this->FTG_frequencies_.setZero(4);
        //this->base_frequency_[0] =  this->env_config_.BASE_FREQUENCY;
        this->FTG_sin_phases_.setZero(4);
        this->FTG_cos_phases_.setZero(4);
        this->FTG_phases_.setZero(4);
        // HISTORICAL DATA
        this->joint_pos_err_hist_.setZero(24);
        this->joint_vel_hist_.setZero(24);
        this->feet_target_hist_.setZero(24);

        this->joint_position_.setZero(12);
        this->joint_velocity_.setZero(12);

        

        // Reward coefficients
        this->rewards_.initializeFromConfigurationFile (cfg["reward"]);

        // Indices of links that should not make contact with ground
        this->foot_indexes_.insert(this->anymal_->getBodyIdx("back_right_lower_leg"));
        this->foot_indexes_.insert(this->anymal_->getBodyIdx("front_right_lower_leg"));
        this->foot_indexes_.insert(this->anymal_->getBodyIdx("back_left_lower_leg"));
        this->foot_indexes_.insert(this->anymal_->getBodyIdx("front_left_lower_leg"));


        std::vector<std::string> feet_link_names = {"front_left_leg_foot", 
                                                    "front_right_leg_foot", 
                                                    "back_left_leg_foot", 
                                                    "back_right_leg_foot"};

        this->height_scanner_ = HeightScanner(this->world_.get(), 
                                        this->anymal_,
                                        cfg["render_height_scan"].template As<bool>(),
                                        feet_link_names,
                                        &this->env_config_);


        // visualize if it is the first environment
        if (this->visualizable_) 
        {
            this->server_ = std::make_unique<raisim::RaisimServer>(this->world_.get());
            this->server_->launchServer(port);
            this->visual_target_ = this->server_->addVisualSphere("goal", 0.2,0, 0, 1, 1);

            this->direction_arrow_body_ = this->server_->addVisualPolyLine("target_dir_line");
            this->direction_arrow_head_ =this->server_->addVisualSphere("direction_arrow_head_", 
                                                                        0.02,1, 0, 0, 1);
            this->turning_arrow_body_ = this->server_->addVisualPolyLine("turning_dir_line");
            this->turning_arrow_head_ = this->server_->addVisualSphere("turning_arrow_head_", 
                                                                            0.02,1, 0, 0, 1);

            // Add color to the poly lines.
            this->direction_arrow_body_->setColor(1, 0, 0, 1);
            this->turning_arrow_body_->setColor(1, 0, 0, 1);
            this->height_scanner_.add_visual_indicators(this->server_.get());
            this->server_->focusOn(this->anymal_);
        }

        // Initialize the time
        this->episode_elapsed_time_ = 0.0;
        this->episode_elapsed_steps_ = 0;

        std::vector<std::string> foot_names, 
                                 thigh_names, 
                                 shank_names;

        thigh_names = {"front_left_hip",
                       "front_right_hip",
                       "back_left_hip",
                       "back_right_hip"};
        
        shank_names = {"front_left_upper_leg",
                       "front_right_upper_leg",
                       "back_left_upper_leg",
                       "back_right_upper_leg"};
        
        foot_names = {"front_left_lower_leg",
                       "front_right_lower_leg",
                       "back_left_lower_leg",
                       "back_right_lower_leg"};
        
                       
        // initialize the contact info solver
        this->contact_solver_ = ContactSolver(this->world_.get(), 
                                               this->anymal_,
                                               this->world_->getTimeStep(),
                                               1.0, // Fricction coefficient mean
                                               0.2, // Fricction coefficient std
                                               thigh_names,
                                               shank_names,
                                               foot_names);
        
        // Initizlize the external force applier
        this->external_force_applier_ = ExternalForceApplier(this->anymal_,
                                                             this->env_config_.EXTERNAL_FORCA_TIME_THRESHOLD,// Aplied force duration [Seconds]
                                                             this->env_config_.EXTERNAL_FORCA_MAX_VALUE// Max force [Newtons]
                                                             );

        // Initiate the random seed
        srand(time(0));
        
        this->external_command_ = false;
        updateTargetPositionAndTurningDirection_();
        updateTargetDirection(); 
        updateVisualTarget_();

        // iterate over the observation parameters and set the values.
        Yaml::Node observation_params = cfg["state_variables"];
        defineObservationSpace_(observation_params);  

        setCommandMode_();  

         
        this->variable_latency_enabled_ = latency_control_params["enabled"].template As<bool>();
        double min_latency = latency_control_params["min"].template As<double>();
        double peak_latency = latency_control_params["peak"].template As<double>();
        double max_latency = latency_control_params["max"].template As<double>();
        std::array<double, 3> i{min_latency, peak_latency, max_latency};
        std::array<double, 3> w{0, 1, 0};
        latency_distribution_ =  std::piecewise_linear_distribution<double>{i.begin(), i.end(), w.begin()};
        
    }
    
    void ENVIRONMENT::setCommandMode_(void){
        // set the command
        if (this->directed_gait_){
            this->current_command_mode_ = COMMAND_MODE_::STRAIGHT;
            this->facing_angle_ = 0;
            return;
        }
        double p = zero_one_real_dist_(this->merssen_twister_);
        if (p < this->straight_prob_){
            this->current_command_mode_ = COMMAND_MODE_::STRAIGHT;
            this->facing_angle_ = 0;
            }
        else if (p < this->straight_prob_ + this->straight_with_random_direction_prob_){
            this->current_command_mode_ = COMMAND_MODE_::STRAIGHT;
            this->facing_angle_ = this->posible_facing_angles_[rand() % this->posible_facing_angles_.size()];
        }
        else if (p < this->straight_prob_ + this->fixed_prob_ + 
                    this->straight_with_random_direction_prob_){
            this->current_command_mode_ = COMMAND_MODE_::FIXED;
            }
        else{
            this->current_command_mode_ = COMMAND_MODE_::STANCE;
            };
        
    };

    void ENVIRONMENT::defineObservationSpace_(Yaml::Node observation_params){
        
        std::map<std::string, int> observation_space_map;

        observation_space_map = {
            {"desired_direction", 
                               this->target_direction_.size()},
            {"desired_turning_direction",
                               1},
            {"body_height",
                               1},
            {"gravity_vector",
                               this->body_rotation_matrix_z_component_.size()},
            {"base_angular_velocity",
                               this->body_angular_vel_.size()},
            {"base_linear_velocity",
                                this->body_linear_vel_.size()},
            {"joint_position",
                               this->joint_position_.size()},
            {"joint_velocity",
                                this->joint_velocity_.size()},
            {"FTG_sin_phases",
                               this->FTG_sin_phases_.size()},
            {"FTG_cos_phases",
                              this->FTG_cos_phases_.size()},
            {"FTG_frequencies",
                               this->FTG_frequencies_.size()},    
            {"base_frequency",
                               1},
            {"joint_pos_err_hist",
                               this->joint_pos_err_hist_.size()},
            {"joint_vel_hist",
                               this->joint_vel_hist_.size()},
            {"feet_target_hist",
                               this->feet_target_hist_.size()},
            {"terrain_normal_at_each_foot",
                               this->contact_solver_.terrain_normal.size()},
            {"feet_height_scan",
                              this->height_scanner_.feet_height_scan.size()},
            {"foot_contact_forces", 
                               this->contact_solver_.foot_contact_forces.size()}, 
            {"foot_contact_states",
                               this->contact_solver_.foot_contact_states.size()},
            {"shank_contact_states",
                               this->contact_solver_.shank_contact_states.size()},    
            {"thigh_contact_states",
                               this->contact_solver_.thigh_contact_states.size()},        
            {"foot_ground_fricction_coeff",
                               this->contact_solver_.foot_ground_friction.size()},  
            {"external_force_applied_to_base",
                               this->external_force_applier_.external_force_base_frame.size()},
            {"pd_constants",
                               this->pd_constants_.size()},
            };
        // vec is a auxiliary variably to order the observation space.
        // We fill the vector with the names of the enabled observations and 
        // the position(order) of the observation in the observation space.
        std::vector<std::pair<std::string, int>> vec;
        int last_idx = 0;
        for (auto const& [key, val] : observation_space_map){
            if (observation_params[key]["enabled"].template As<bool>()){
                int order;
                if (observation_params[key]["priviledge"].template  As<bool>()){
                    // If the observation is priviledged, it is placed at the end of the observation space.
                    order = observation_params[key]["position"].template As<int>() + 10000;
                }
                else {
                    order = observation_params[key]["position"].template As<int>();
                };
                vec.push_back(std::make_pair(key, order));
                last_idx += val;
            }       
        } 
        // We sort the vector by the position order.
        std::sort(vec.begin(), vec.end(),
            [](const std::pair<std::string, int> &l, const std::pair<std::string, int> &r)
            {
                if (l.second != r.second) {
                    return l.second < r.second;
                }
                return l.first < r.first;
            });
        // Now iterate over the map and define the ob_idx_dict_
        
        // Define the observarion dimension and the observation space
        this->obDim_ = last_idx;
        this->ob_double_.setZero(this->obDim_); 
        this->noise_scaler_vec_.setZero(this->obDim_);

        last_idx = 0;
        int vec_size;
        for (auto const& item : vec){
            vec_size = observation_space_map[item.first];
            this->observations_idx_map_[item.first] = {last_idx, vec_size};
            this->ob_idx_dict_[item.first] = {last_idx, last_idx + vec_size}; 
            double noise_value = observation_params[item.first]["noise_std"].template As<double>();
            for (int i = 0; i < vec_size; i++){
                this->noise_scaler_vec_(last_idx + i) = noise_value;
            }
            last_idx += vec_size;                 
        }
        this->enable_noise_ = true;
        if (this->noise_scaler_vec_ == Eigen::VectorXd::Zero(this->obDim_)){
            this->enable_noise_ = false;
        }
        // Temporary fix, until the priviledge observation space is properly defined in the config file.
        this->ob_idx_dict_["encoder_obs"] = {0, 49};
        this->ob_idx_dict_["non_priviliged"] = {0, 121};
        this->ob_idx_dict_["priviliged"] = {121, 368};
    }

    void ENVIRONMENT::updateVisualTarget_(void) 
    {   
        if (this->visualizable_) 
        {
            auto& visual_target_height_rt = this->world_->rayTest(
                {this->target_position_[0],this->target_position_[1],10}, 
                {this->target_position_[0],this->target_position_[1],-1}, 
                50.0, 
                true
            );
            this->visual_target_->setPosition(
                this->target_position_[0],
                this->target_position_[1],
                visual_target_height_rt[0].getPosition()[2]
            );
        };
    }

    void ENVIRONMENT::updateTargetDirection(void) 
    {
        
        // If directed_gait_ is true, change the turning_direction_ to reduce 
        // the target_direction_angle_ to 0.
        if (this->current_command_mode_ == COMMAND_MODE_::STRAIGHT) 
        {
            this->target_direction_ = (this->target_position_ - this->gc_.head(2)).normalized();
            double oldX = this->target_direction_[0];
            double oldY = this->target_direction_[1];
            
            // Rotate the target direction to the robot base frame.

            double angle = -base_euler_[2];
            this->target_direction_[0] = oldX * std::cos(angle) -  oldY * std::sin(angle);
            this->target_direction_[1] = oldX * std::sin(angle) +  oldY * std::cos(angle);
            
            this->target_direction_angle_ = std::atan2(
                this->target_direction_[1], 
                this->target_direction_[0]
            );
            if (std::abs(this->target_direction_angle_ - this->facing_angle_) > M_PI/6) 
            {
                this->turning_direction_ = (this->target_direction_angle_ - this->facing_angle_)/
                    std::abs(this->target_direction_angle_ - this->facing_angle_);
            } 
            else 
            {
                this->turning_direction_ = 0;
            };
        }
    }

    void ENVIRONMENT::updateTargetPositionAndTurningDirection_(void)
    {   
        if (this->current_command_mode_ == COMMAND_MODE_::STRAIGHT ||
            this->current_command_mode_ == COMMAND_MODE_::FIXED){
            this->env_config_.CARTESIAN_DELTA = true;
            this->turning_direction_ = 0;
            
            this->target_direction_angle_ = -M_PI + rand() * (M_PI + M_PI) / RAND_MAX;
            this->target_position_[0] = gc_[0] + 1.5 * std::cos(this->target_direction_angle_);
            if (this->terrain_type_ == TERRAIN_TYPES_::STAIRS)
            {
                this->target_position_[1] = rand() % 100 > 50 ? 10.0 : -10.0;
            }
            else 
            {
                this->target_position_[1] = gc_[1] + 1.5 * std::sin(this->target_direction_angle_);
            }

            // Saturate the target_position_ to scale of the terrain.
            if (std::abs(this->target_position_[0]) > 0.9*this->terrain_x_size_/2){
                this->target_position_[0] = 0.9 * this->target_position_[0] / std::abs(this->target_position_[0]) * this->terrain_x_size_/2;
            }
            if (std::abs(this->target_position_[1]) > 0.9*this->terrain_y_size_/2){
                this->target_position_[1] = 0.9*this->target_position_[1] / std::abs(this->target_position_[1]) * this->terrain_y_size_/2;
            }

            this->target_direction_ = (this->target_position_ - this->gc_.head(2)).normalized();
            
        }
        else if (this->current_command_mode_ == COMMAND_MODE_::STANCE){
            this->target_position_.setZero(2);
            this->target_direction_.setZero(2);
            this->target_direction_angle_ = 0;
            this->turning_direction_ = this->minus_one_one_dist(this->merssen_twister_);
            this->env_config_.CARTESIAN_DELTA = false;
        }
    }

    void ENVIRONMENT::updateVisualTargetDirection_() {
        if (this->visualizable_) {
            double scale = 0.3;
            this->direction_arrow_body_-> clearPoints();
            this->direction_arrow_body_->addPoint(this->gc_.head(3) + Eigen::Vector3d(0,0,0.35));
            Eigen::Vector2d tar_dir = (this->target_position_ - this->gc_.head(2)).normalized();
            Eigen::Vector3d direction_arrow_head_pos(this->gc_[0] + tar_dir[0] * scale, 
                                                    this->gc_[1] + tar_dir[1]* scale, 
                                                        this->gc_[2] + 0.35);
            this->direction_arrow_body_->addPoint(direction_arrow_head_pos);
            this->direction_arrow_head_->setPosition(direction_arrow_head_pos[0],
                                                direction_arrow_head_pos[1],
                                                direction_arrow_head_pos[2]);

            this->turning_arrow_body_-> clearPoints();
            this->turning_arrow_head_->setPosition(0, 0, 100);
            if (this->turning_direction_){
                this->turning_arrow_body_->addPoint(this->gc_.head(3) + Eigen::Vector3d(0,0,0.35));
                
                Eigen::Vector3d turning_arrow_head_pos(
                    this->gc_[0] + 0 * this->turning_direction_ * scale, 
                    this->gc_[1] + 0 * this->turning_direction_ * scale, 
                    this->gc_[2] + 1 * this->turning_direction_ * scale + 0.35);
                
                this->turning_arrow_body_->addPoint(turning_arrow_head_pos);
                
                this->turning_arrow_head_->setPosition(turning_arrow_head_pos[0], 
                                                       turning_arrow_head_pos[1], 
                                                       turning_arrow_head_pos[2]);
            }
        };
    }

    void ENVIRONMENT::reset(void)
    {
        // We perform a raycast to get the height of the ground around 
        // the x = 0, y = 0. This is used to set the initial height of the 
        // robot. This is important because the robot is not standing on 
        // the ground when it is created.
        const raisim::RayCollisionList& col = this->world_->rayTest(
            {0, 0, 10}, 
            {0, 0, -1}, 
            50.0, 
            true
        );
        double z_init = 0.25;
        if (col.size() > 0)
        {
            z_init += col[0].getPosition()[2];
        }

        // This is nominal configuration of anymal
        // Configuracion: [x, y, z, qr, qi, qj, qk, 12 joints angles]
        this->gc_init_[2] = z_init;
        this->anymal_->setState(this->gc_init_, this->gv_init_);

        // Reset the command angle (an angle from -pi to pi)
        // Reset the target turning direction as a random number between -1 and 1 
        this->episode_elapsed_time_ = 0.0;
        this->episode_elapsed_steps_ = 0;
        this->traverability_ = 0.0;

        // Re initialize external force applier
        this->external_force_applier_ = ExternalForceApplier(
            this->anymal_,
            this->env_config_.EXTERNAL_FORCA_TIME_THRESHOLD,
            this->env_config_.EXTERNAL_FORCA_MAX_VALUE
        );
        this->external_command_ = false;

        setPDGains_();
        
        
        setCommandMode_();
        updateObservation_();
        
        updateTargetPositionAndTurningDirection_();
        updateTargetDirection();
        
        
        updateVisualTarget_();

        
        
    }

    float ENVIRONMENT::step(const Eigen::Ref<EigenVec>& action)
    {
        // Action scaling
        this->action_scaled_ = action.cast<double>();
        // print the action
        this->action_scaled_ = action_scaled_.cwiseProduct(this->actionScale_) + 
            this->action_mean_;
        
        if (!this->external_command_){
            updateTargetDirection();
        }
        

        // Update the historical data
        this->joint_pos_err_hist_.tail(12) = this->joint_pos_err_hist_.head(12);
        this->joint_vel_hist_.tail(12)     = this->joint_vel_hist_.head(12);
        this->feet_target_hist_.tail(12)   = this->feet_target_hist_.head(12);


        // We must use the observation vector as the noise was added to it not to the joint angles
        // vector during the updateObservation_() function
        //this->observations_idx_map_[item.first] = {last_idx, vec_size};
        std::vector<int> ob_seg_idx = this->observations_idx_map_["joint_position"];
        this->joint_pos_err_hist_.head(12) = this->ob_double_.segment(ob_seg_idx[0], ob_seg_idx[1]) - 
                                             this->joints_target_;
        
        ob_seg_idx = this->observations_idx_map_["joint_velocity"];
        this->joint_vel_hist_.head(12) = this->ob_double_.segment(ob_seg_idx[0], ob_seg_idx[1]);
        // This is ok because the target history is not a noisy observation
        this->feet_target_hist_.head(12) = this->feet_target_pos_ ;


        // print the action scaled
        std::tuple<Eigen::VectorXd, 
                   Eigen::VectorXd, 
                   Eigen::Vector4d, 
                   Eigen::Vector4d, 
                   Eigen::Vector4d,
                   Eigen::Vector4d> control_pipeline_output;

        control_pipeline_output = controlPipeline(
                                                this->action_scaled_,
                                                this->turning_direction_,
                                                this->target_direction_angle_,
                                                this->base_euler_(0), 
                                                this->base_euler_(1), 
                                                this->episode_elapsed_time_,
                                                &this->env_config_);
        // print the control pipeline input and output
        this->joints_target_    = std::get<0>(control_pipeline_output);
        this->feet_target_pos_  = std::get<1>(control_pipeline_output);
        this->FTG_frequencies_  = std::get<2>(control_pipeline_output);
        this->FTG_sin_phases_   = std::get<3>(control_pipeline_output);
        this->FTG_cos_phases_   = std::get<4>(control_pipeline_output);
        this->FTG_phases_       = std::get<5>(control_pipeline_output);

        this->p_target_.tail(this->n_joints_) = this->joints_target_;
        this->anymal_->setPdTarget(this->p_target_, this->v_target_);
        
        Eigen::VectorXd joint_vel, joint_vel_prev;
        joint_vel.setZero(this->n_joints_);
        joint_vel_prev.setZero(this->n_joints_);
        joint_vel_prev = this->gv_.tail(12);


        int sim_steps;
        // if the latency modifier is enabled we are forced to modify the 
        if (this->variable_latency_enabled_){
            double latency = this->latency_distribution_(this->random_gen_);
            this->control_dt_ = 1/latency;
            this->env_config_.CONTROL_DT = this->control_dt_;

            sim_steps = int(control_dt_ / simulation_dt_ + 1e-10);

            double dt = this->control_dt_ / sim_steps;
            
            this->world_->setTimeStep(dt);
        }
        
        else{
            sim_steps = int(this->control_dt_ / simulation_dt_ + 1e-10);
        }
        
        for(int i = 0; i < sim_steps; i++)
        {   
            this->external_force_applier_.apply_external_force(this->episode_elapsed_time_);
            if(this->server_) this->server_->lockVisualizationServerMutex();
            this->world_->integrate();
            if(this->server_) this->server_->unlockVisualizationServerMutex();
            
            
        }
        this->anymal_->getState(this->gc_, this->gv_);
        joint_vel = this->gv_.tail(12);
        this->joint_acceleration_ = (joint_vel - joint_vel_prev) / this->control_dt_;

        // Traversability calculation
        int traverability = (this->body_linear_vel_[0] * this->target_direction_[0] +
            this->body_linear_vel_[1] * this->target_direction_[1]) >= MIN_DESIRED_VEL;
        this->traverability_ = (this->episode_elapsed_steps_ * this->traverability_ +
            traverability) / (this->episode_elapsed_steps_ + 1);

        // Step the time
        this->episode_elapsed_steps_ += 1;
        // this->episode_elapsed_time_ = this->episode_elapsed_steps_ * this->control_dt_;
        this->episode_elapsed_time_ += this->control_dt_;
        // Limit the  elpapsed time to the episode duration using the mod function
        this->episode_elapsed_time_ = fmod(this->episode_elapsed_time_, this->episode_duration_);

        // Check if the robot is near the goal visual object

        if (!this->external_command_){
            if ( (this->current_command_mode_ == COMMAND_MODE_::FIXED || 
              this->current_command_mode_ == COMMAND_MODE_::STRAIGHT)
             && (this->target_position_ - this->gc_.head(2)).norm() < 0.2)
            {
                updateTargetPositionAndTurningDirection_();
                updateVisualTarget_();
            }
            else if ((this->current_command_mode_ == COMMAND_MODE_::STANCE ||
                    this->current_command_mode_ == COMMAND_MODE_::FIXED)   && 
                    this->episode_elapsed_time_ >= this->command_duration_){
                this->command_duration_ += this->command_duration_;
                updateTargetPositionAndTurningDirection_();
                updateVisualTarget_();
            }
        }
        
        // Update observations
        updateObservation_();

        // Rewads
        registerRewards();

        // Update other visual elements
        updateVisualTargetDirection_();

        return this->rewards_.sum();
    }    

    void ENVIRONMENT::registerRewards(void)
    {
        this->proj_linear_vel_  = this->target_direction_.dot(this->body_linear_vel_.head(2));
        this->proj_angular_vel_ = this->turning_direction_ * this->body_angular_vel_[2];
        // -------------------------------------------------------------------//
        // Torque reward
        // -------------------------------------------------------------------//
        this->torque_reward_ = this->anymal_->getGeneralizedForce().e().tail(12).squaredNorm();
        this->rewards_.record("torque", this->curriculum_coeff_ * this->torque_reward_);
        // -------------------------------------------------------------------//
        // Linear Velocity Reward
        // -------------------------------------------------------------------//
        if (this->target_direction_ == Eigen::Vector2d::Zero()){
            this->linear_vel_reward_ = std::exp(-1.5 *std::pow(this->proj_linear_vel_, 2) );;
        }
        else{  
            if (this->proj_linear_vel_ < this->env_config_.VEL_TH)
            {
                this->linear_vel_reward_ = std::exp(
                    -1.5 * std::pow(this->proj_linear_vel_ - this->env_config_.VEL_TH, 2) 
                );
            }
            else
            {
                this->linear_vel_reward_ = 1.0;
            };
        }
        this->rewards_.record("linearVel", linear_vel_reward_);
        // -------------------------------------------------------------------//
        // Angluar Velocity Reward
        // -------------------------------------------------------------------//
        // This reward should only be used when the target direction is not 0.
        if (this->turning_direction_){ 
            if (this->proj_angular_vel_ < this->env_config_.VEL_TH){
                this->angular_vel_reward_ = std::exp(
                    -1.5 * std::pow(this->proj_angular_vel_ - this->env_config_.VEL_TH, 2) 
                );
            }
            else
            {
                this->angular_vel_reward_ = 1.0;
            };
        }
        else
        {
           this->angular_vel_reward_ = std::exp(-1.5 *std::pow(this->proj_angular_vel_, 2) ); 
        };
        this->rewards_.record("angularVel", angular_vel_reward_);
        // -------------------------------------------------------------------//
        // Base motion reward
        // -------------------------------------------------------------------//
        this->h_linear_vel_ = this->body_linear_vel_.head(2);
        this->ort_vel_ = (this->h_linear_vel_ - this->target_direction_ * this->proj_linear_vel_).norm();
        
        this->h_angular_vel_ = this->body_angular_vel_.head(2);
        this->w_2_ = this->h_angular_vel_.dot(this->h_angular_vel_);
        this->base_motion_reward_ = std::exp(
            -1.5 * std::pow(this->ort_vel_, 2)) + std::exp(-1.5 * this->w_2_
        );
        rewards_.record("baseMotion", base_motion_reward_);

        // -------------------------------------------------------------------//
        // Body motion Reward:
        // penalizes the body velocity in directions not part of the command
        // -------------------------------------------------------------------//
        double v_z = this->body_linear_vel_(2);
        //double w_x = this->body_angular_vel_(0);
        //double w_y = this->body_angular_vel_(1);
        //double body_motion_reward =  - 1.25 *  std::pow(v_z, 2) -  0.4 * std::abs(w_x) -  0.4 * std::abs(w_y); 
        double body_motion_reward;
        body_motion_reward = std::exp(-1.5 * std::pow(v_z, 2)) + std::exp(-1.5 * this->w_2_);
        rewards_.record("bodyMotion", body_motion_reward);
        
        // -------------------------------------------------------------------//
        // Linear Orthogonal Velocity Reward:
        // penalizes the velocity orthogonal to the target direction
        // -------------------------------------------------------------------//
        double linear_orthogonal_vel_reward;

        this->h_linear_vel_ = this->body_linear_vel_.head(2);
        this->ort_vel_ = (this->h_linear_vel_ - this->target_direction_ * this->proj_linear_vel_).norm();

        linear_orthogonal_vel_reward = std::exp(
            -3 * std::pow(this->ort_vel_, 2)
        );
        rewards_.record("linearOrthogonalVelocity", linear_orthogonal_vel_reward);

        // -------------------------------------------------------------------//
        // Body collision reward
        // ---------------------
        // Penalizes undesirable collisions between the robot and the 
        // environment.
        // Collisions between articulations that are not the robot feet are 
        // penalized.
        // -------------------------------------------------------------------//
        this->rewards_.record("bodyCollision", 
                      this->curriculum_coeff_ * this->contact_solver_.undesirable_collisions_reward_);
        // -------------------------------------------------------------------//
        // Foot Clearance reward
        // ---------------------
        // When a leg is in swing phase, the robot should lift the corresponding
        // foot higher than the surroundings to avoid collision
        // -------------------------------------------------------------------//
        this->rewards_.record(
            "footClearance", 
            this->height_scanner_.calculate_foot_clearance_reward(this->FTG_phases_)
        );
        
        // -------------------------------------------------------------------//
        // Target Smoothness reward
        // ---------------------
        // The magnitude of the second order finite difference derivatives of 
        // the target foot positions are penalized such that the generated foot 
        // trajectories become smoother.
        // We also added the magnitude of the first order derivative.
        // -------------------------------------------------------------------//
        double target_smoothness_reward ;
        double tgs_1st, tgs_2nd;
        tgs_1st = (this->feet_target_pos_ - this->feet_target_hist_.head(12)).norm();
        tgs_2nd = (this->feet_target_pos_ - 2 * this->feet_target_hist_.head(12) + this->feet_target_hist_.tail(12)).norm(); 

        target_smoothness_reward = - (tgs_1st + tgs_2nd);
        this->rewards_.record("targetSmoothness", 
                       this->curriculum_coeff_ * target_smoothness_reward);

        // -------------------------------------------------------------------//
        // Joint Motion Reward
        // ---------------------
        // Penalizes joint velocity and acceleration to avoid vibrations:.
        // -------------------------------------------------------------------//

        double joint_motion_reward;
        joint_motion_reward = -(0.01 * this->gv_.tail(12).squaredNorm() + this->joint_acceleration_.squaredNorm());

        this->rewards_.record("jointMotion", this->curriculum_coeff_ * joint_motion_reward);  


        // -------------------------------------------------------------------//
        // Slip Reward
        // ---------------------
        // Penalizes the foot velocity if the foot is in contact with the ground to reduce slippage.
        // -------------------------------------------------------------------//

        double slip_reward;
        slip_reward = -(this->height_scanner_.feet_speed_squared_.dot(this->contact_solver_.foot_contact_states_));

        this->rewards_.record("slip", this->curriculum_coeff_ * slip_reward); 

    }

    void ENVIRONMENT::updateObservation_(void) 
    {
        this->anymal_->getState(this->gc_, this->gv_);
        raisim::Vec<4> quat;
        raisim::Mat<3,3> rot;
        quat[0] = this->gc_[3]; 
        quat[1] = this->gc_[4]; 
        quat[2] = this->gc_[5]; 
        quat[3] = this->gc_[6];

        raisim::quatToRotMat(quat, rot);
        this->body_linear_vel_  = rot.e().transpose() * this->gv_.segment(0, 3);
        this->body_angular_vel_ = rot.e().transpose() * this->gv_.segment(3, 3);

        Quaternion bd_quat;
        bd_quat.w = this->gc_[3]; 
        bd_quat.x = this->gc_[4]; 
        bd_quat.y = this->gc_[5]; 
        bd_quat.z = this->gc_[6];
        EulerAngles body_orientation;
        body_orientation = ToEulerAngles(bd_quat);
        

        /// Body orientation in euler angles
        this->base_euler_ <<   body_orientation.roll  + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
                                    body_orientation.pitch + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
                                    body_orientation.yaw   + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_;
        
        this->height_scanner_.calculate_foot_scan(body_orientation.yaw);
        
        //this->body_rotation_matrix_z_component_ = rot.e().row(2).transpose();
        // This way the noise from the orientation is propagated to the gravity vector
        this->body_rotation_matrix_z_component_ << - std::sin(body_orientation.pitch),
                                   std::sin(body_orientation.roll) * std::cos(body_orientation.pitch),
                                   std::cos(body_orientation.roll) * std::cos(body_orientation.pitch);



        auto& height_rt = world_->rayTest(
                {gc_[0],gc_[1],  gc_[2]}, 
                {-this->body_rotation_matrix_z_component_[0],
                 -this->body_rotation_matrix_z_component_[1],
                 -this->body_rotation_matrix_z_component_[2]}, 
                 50., 
                 true, 0, 0, raisim::RAISIM_STATIC_COLLISION_GROUP);

        this->body_height_ = (gc_.head(3) - height_rt[0].getPosition()).norm();
        //this->body_height_vec_[0] = this->body_height_;
 
        this->contact_solver_.compute_contact_info();

        this->joint_position_ = this->gc_.tail(12);
        this->joint_velocity_ = this->gv_.tail(12);

        this->external_force_applier_.calculate_external_force_in_base_frame(rot.e());

        for (auto const& [key, val] : this->observations_idx_map_){
            if (key == "desired_direction"){
                this->ob_double_.segment(val[0], val[1]) =  this->target_direction_;
            }
            else if (key == "desired_turning_direction"){
                this->ob_double_(val[0], val[1]) = this->turning_direction_;
            }                    
            
            else if (key == "body_height"){
                this->ob_double_(val[0], val[1]) = this->body_height_;
            }
            
            else if (key == "gravity_vector"){
                this->ob_double_.segment(val[0], val[1]) = this->body_rotation_matrix_z_component_;
            }
                        
            else if (key == "base_angular_velocity"){
                this->ob_double_.segment(val[0], val[1]) = this->body_angular_vel_;
            }
                
            else if (key == "base_linear_velocity"){
                this->ob_double_.segment(val[0], val[1]) = this->body_linear_vel_;
            }
                    
            else if (key == "joint_position"){
                this->ob_double_.segment(val[0], val[1]) = this->joint_position_;
            }
                        
            else if (key == "joint_velocity"){
                this->ob_double_.segment(val[0], val[1]) = this->joint_velocity_;
            }
                        
            else if (key == "FTG_sin_phases"){
                this->ob_double_.segment(val[0], val[1]) = this->FTG_sin_phases_;
            }
                        
            else if (key == "FTG_cos_phases"){
                this->ob_double_.segment(val[0], val[1]) = this->FTG_cos_phases_;    
            }
                        
            else if (key == "FTG_frequencies"){
                this->ob_double_.segment(val[0], val[1]) = this->FTG_frequencies_;
            }
                        
            else if (key == "base_frequency"){
                this->ob_double_(val[0], val[1]) =  this->env_config_.BASE_FREQUENCY;
            }
                        
            else if (key == "joint_pos_err_hist"){
                this->ob_double_.segment(val[0], val[1]) = this->joint_pos_err_hist_;
            }
                    
            else if (key == "joint_vel_hist"){
                this->ob_double_.segment(val[0], val[1]) = this->joint_vel_hist_;
            }
                    
            else if (key == "feet_target_hist"){
                this->ob_double_.segment(val[0], val[1]) = this->feet_target_hist_;  
            }
                        
            else if (key == "terrain_normal_at_each_foot"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.terrain_normal_at_each_foot_;
            }
            
            else if (key == "feet_height_scan"){
                this->ob_double_.segment(val[0], val[1]) = this->height_scanner_.feet_height_scan_;
            }
                        
            else if (key == "foot_contact_forces"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.foot_contact_forces_;
            } 
                    
            else if (key == "foot_contact_states"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.foot_contact_states_;
            }
                    
            else if (key == "shank_contact_states"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.shank_contact_states_;
            }
                    
            else if (key == "thigh_contact_states"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.thigh_contact_states_;
            }
                    
            else if (key == "foot_ground_fricction_coeff"){
                this->ob_double_.segment(val[0], val[1]) = this->contact_solver_.foot_ground_friction_coeff_;
            }
            
            else if (key == "external_force_applied_to_base"){
                this->ob_double_.segment(val[0], val[1]) = this->external_force_applier_.external_force_base_frame_;
            }

            else if (key == "pd_constants"){
                this->ob_double_.segment(val[0], val[1]) = this->pd_constants_;
            }

        }

        // Noise addition
        if (this->enable_noise_){
            Eigen::VectorXd noise_vec = Eigen::VectorXd::NullaryExpr(this->obDim_,[&](){ return this->norm_dist_(this->random_gen_); });
            noise_vec = this->curriculum_coeff_ * noise_vec;
            this->ob_double_ += this->noise_scaler_vec_ * noise_vec;
        }

    }

    void ENVIRONMENT::observe(Eigen::Ref<EigenVec> ob) 
    {
        // Convert it to float
        ob = this->ob_double_.cast<float>();
    }

    void ENVIRONMENT::updateBaseEulerAngles_(Eigen::Ref<EigenVec> ea) 
    {
        // convert it to float
        ea = this->base_euler_.cast<float>();
    }

    bool ENVIRONMENT::isTerminalState(void) 
    {
        // Get the contact states 
        // if none of its four feet are in contact with the ground then the state is terminal'
        // We might have to change this to be more general

        if (this->body_rotation_matrix_z_component_[2] < 0.6){
            for(auto& contact: this->anymal_->getContacts())
        {
            if (
                this->foot_indexes_.find(contact.getlocalBodyIndex()) == 
                this->foot_indexes_.end()
            )
            {
                return true; 
            }
        }
        }

        return false;
    }

    void ENVIRONMENT::hills_(
        double frequency, 
        double amplitude, 
        double roughness
    ) {
        this->generator.clear();
        this->generator.hills(frequency, amplitude, roughness);
        this->terrain_type_ = TERRAIN_TYPES_::HILLS;
        this->terrain_x_size_ = this->generator.terrain_x_size;
        this->terrain_y_size_ = this->generator.terrain_y_size;
        
    }

    void ENVIRONMENT::stairs(double width, double height)
    {
        this->generator.clear();
        this->generator.stairs(width, height);
        this->terrain_type_ = TERRAIN_TYPES_::STAIRS;
        this->terrain_x_size_ = this->generator.terrain_x_size;
        this->terrain_y_size_ = this->generator.terrain_y_size;
    }

    void ENVIRONMENT::cellularSteps(double frequency, double amplitude)
    {
        this->generator.clear();
        this->generator.cellular_steps(frequency,  amplitude);
        this->terrain_type_ = TERRAIN_TYPES_::CELULAR_STEPS;
        this->terrain_x_size_ = this->generator.terrain_x_size;
        this->terrain_y_size_ = this->generator.terrain_y_size;
    }

    void ENVIRONMENT::steps(double widht, double height)
    {
        this->generator.clear();
        this->generator.steps(widht,  height);
        this->terrain_type_ = TERRAIN_TYPES_::STEPS;
        this->terrain_x_size_ = this->generator.terrain_x_size;
        this->terrain_y_size_ = this->generator.terrain_y_size;
    }

    void ENVIRONMENT::slope(double slope, double roughness)
    {
        this->generator.clear();
        this->generator.slope(slope, roughness);
        this->terrain_type_ = TERRAIN_TYPES_::SLOPE;
        this->terrain_x_size_ = this->generator.terrain_x_size;
        this->terrain_y_size_ = this->generator.terrain_y_size;
    }

    double ENVIRONMENT::traversability_(void)
    {
        return this->traverability_;
    }

    double ENVIRONMENT::getPower(void){
        Eigen::VectorXd torque; 
        torque = - 50 * (this->joint_position_ - this->joints_target_) - 0.2 * this->joint_velocity_;
        torque = torque.cwiseMin(4.0).cwiseMax(-4.0);
        return torque.dot(this->joint_velocity_);
        
    };

    double ENVIRONMENT::getFroude(void){
        double g = this->world_->getGravity().e().norm();
        double v2 = this->body_linear_vel_.head(2).squaredNorm();
        return v2 / (g * this->body_height_);
    };

    double ENVIRONMENT::getProjSpeed(void){
        return this->proj_linear_vel_;
    };

    double ENVIRONMENT::getMaxTorque(void){
        Eigen::VectorXd torque;
        torque = - 50 * (this->joint_position_ - this->joints_target_) - 0.2 * this->joint_velocity_;
        torque = torque.cwiseMin(4.0).cwiseMax(-4.0);
        return torque.maxCoeff();
    };

     void ENVIRONMENT::setCommand(double direction_angle, double turning_direction, bool stop){

        this->target_direction_angle_ = direction_angle;
        this->turning_direction_ =  turning_direction;

        if (stop){
            this->target_direction_ = Eigen::Vector2d::Zero();
            this->env_config_.CARTESIAN_DELTA = false;
        }
        else{
            this->target_direction_ << std::cos(direction_angle), std::sin(direction_angle);
            this->env_config_.CARTESIAN_DELTA = true;
        }
        
          // - this->gc_.head(2)

        this->external_command_ = true;

        raisim::Vec<4> quat;
        raisim::Mat<3,3> rot;
        quat[0] = this->gc_[3]; 
        quat[1] = this->gc_[4]; 
        quat[2] = this->gc_[5]; 
        quat[3] = this->gc_[6];

        raisim::quatToRotMat(quat, rot);

        this->target_position_ = rot.e().block(0,0,2,2).transpose() * this->target_direction_;
        // print the turning direction
        
   }

   void ENVIRONMENT::setPDGains_(){
        // Calculate the PD from the current state
        Eigen::VectorXd p_vec = Eigen::VectorXd::NullaryExpr(
            this->n_joints_,
            [&]() { 
                double x = zero_one_real_dist_(this->merssen_twister_);
                return this->p_min_ + x * (this->p_max_ - this->p_min_ );
            }
        );
        Eigen::VectorXd d_vec = Eigen::VectorXd::NullaryExpr(
            this->n_joints_,
            [&]() { 
                double x = zero_one_real_dist_(this->merssen_twister_);
                return this->d_min_ + x * (this->d_max_ - this->d_min_ );
            }
        );
        Eigen::VectorXd jointPgain(this->gv_dim_), jointDgain(this->gv_dim_);
        
        jointPgain.setZero(); 
        jointDgain.setZero(); 
        jointPgain.tail(this->n_joints_) << p_vec;
        jointDgain.tail(this->n_joints_) << d_vec;
        
        this->pd_constants_.head(this->n_joints_) << jointPgain.tail(this->n_joints_);
        this->pd_constants_.tail(this->n_joints_) << jointDgain.tail(this->n_joints_);
        this->anymal_->setPdGains_(jointPgain, jointDgain);
   }
}