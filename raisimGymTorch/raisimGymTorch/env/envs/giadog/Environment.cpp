#include "Environment.hpp"

namespace raisim
{
    ENVIRONMENT::ENVIRONMENT(
        const std::string &resource_dir,
        const Yaml::Node &cfg,
        bool visualizable,
        int port) : RaisimGymEnv(resource_dir, cfg, port),
                    visualizable_(visualizable),
                    norm_dist_(0, 1),
                    port_(port)
    {
        // Get config values
        this->env_config_.H = cfg["gait"]["foot_vertical_span"].template As<double>();
        this->env_config_.SIGMA_0[0] = cfg["gait"]["leg_1_phase"].template As<double>();
        this->env_config_.SIGMA_0[1] = cfg["gait"]["leg_2_phase"].template As<double>();
        this->env_config_.SIGMA_0[2] = cfg["gait"]["leg_3_phase"].template As<double>();
        this->env_config_.SIGMA_0[3] = cfg["gait"]["leg_4_phase"].template As<double>();
        this->env_config_.ANGULAR_DELTA   = cfg["gait"]["angular_movement_heuristic"].template As<bool>();
        this->env_config_.BASE_FREQUENCY  = cfg["gait"]["base_frequency"].template As<double>();
        this->env_config_.CARTESIAN_DELTA = cfg["gait"]["cartesian_movement_heuristic"].template As<bool>();
        this->env_config_.USE_HORIZONTAL_FRAME = cfg["gait"]["use_horizontal_frame"].template As<bool>();
        
        this->env_config_.X_MOV_DELTA = cfg["gait"]["x_movement_delta"].template As<double>();
        this->env_config_.Y_MOV_DELTA = cfg["gait"]["y_movement_delta"].template As<double>();
        this->env_config_.ANG_MOV_DELTA = cfg["gait"]["angular_movement_delta"].template As<double>();

        this->env_config_.H_OFF     = cfg["robot"]["h_off"].template As<double>();
        this->env_config_.V_OFF     = cfg["robot"]["v_off"].template As<double>();
        this->env_config_.LEG_SPAN  = cfg["robot"]["leg_span"].template As<double>();
        this->env_config_.THIGH_LEN = cfg["robot"]["thigh_len"].template As<double>();
        this->env_config_.SHANK_LEN = cfg["robot"]["shank_len"].template As<double>();
        
        
        this->p_max_ = cfg["robot"]["pd_gains"]["kp_max"].template As<double>();
        this->p_min_ = cfg["robot"]["pd_gains"]["kp_min"].template As<double>();
        this->d_max_ = cfg["robot"]["pd_gains"]["kd_max"].template As<double>();
        this->d_min_ = cfg["robot"]["pd_gains"]["kd_min"].template As<double>();
        

        this->noise_ = cfg["simulation"]["noise"].template As<bool>();
        this->simulation_dt_ = cfg["simulation"]["simulation_dt"].template As<double>();
        this->episode_duration_ = cfg["simulation"]["episode_max_time"].template As<double>();
        this->variable_latency_ = cfg["simulation"]["latency"]["variable"].template As<bool>();
        this->env_config_.N_SCAN_RINGS = cfg["simulation"]["height_scan"]["n_scan_rings"].template As<int>();
        this->env_config_.SCANS_PER_RING = cfg["simulation"]["height_scan"]["scans_per_ring"].template As<int>();
        this->env_config_.FOOT_SCAN_RADIUS = cfg["simulation"]["height_scan"]["foot_scan_radius"].template As<double>();
        this->orientation_noise_std_ = cfg["simulation"]["noise_std"]["orientation"].template As<double>() * this->noise_;

        this->observations_noise_["linear_velocity"] = cfg["simulation"]["noise_std"]["linear_velocity"].template As<double>() * this->noise_;
        this->observations_noise_["angular_velocity"] = cfg["simulation"]["noise_std"]["angular_velocity"].template As<double>() * this->noise_;
        this->observations_noise_["joint_position"] = cfg["simulation"]["noise_std"]["joint_position"].template As<double>() * this->noise_;
        this->observations_noise_["joint_velocity"] = cfg["simulation"]["noise_std"]["joint_velocity"].template As<double>() * this->noise_;
        
        // print the noise values
        //RSINFO("Orientation noise: " << this->orientation_noise_std_);
        //RSINFO("Linear velocity noise: " << this->observations_noise_["linear_velocity"]);
        //RSINFO("Angular velocity noise: " << this->observations_noise_["angular_velocity"]);
        //RSINFO("Joint position noise: " << this->observations_noise_["joint_position"]);
        //RSINFO("Joint velocity noise: " << this->observations_noise_["joint_velocity"]);


        this->display_target_ = cfg["simulation"]["display"]["target"].template As<bool>();
        this->display_height_ = cfg["simulation"]["display"]["height"].template As<bool>();
        this->display_turning_ = cfg["simulation"]["display"]["turning"].template As<bool>();
        this->display_direction_ = cfg["simulation"]["display"]["direction"].template As<bool>();
        this->display_linear_vel_ = cfg["simulation"]["display"]["linear_vel"].template As<bool>();
        this->display_x_component_ = cfg["simulation"]["display"]["x_component"].template As<bool>();
        this->display_y_component_ = cfg["simulation"]["display"]["y_component"].template As<bool>();
        this->display_z_component_ = cfg["simulation"]["display"]["z_component"].template As<bool>();
        this->display_angular_vel_ = cfg["simulation"]["display"]["angular_vel"].template As<bool>();

        this->env_config_.VEL_TH = cfg["train"]["reward"]["velocity_threshold"].template As<double>();
        this->env_config_.MAX_EXTERNAL_FORCE = cfg["train"]["max_external_force"].template As<double>();
        this->env_config_.EXTERNAL_FORCE_TIME = cfg["train"]["external_force_time"].template As<double>();
        int curriculum_grows_start_ = cfg["train"]["curriculum_grows_start"].template As<int>();
        int curriculum_grows_duration_ = cfg["train"]["curriculum_grows_duration"].template As<int>();

        this->traversability_min_speed_treshold_ = cfg["train"]["traversability_min_speed_treshold"].template As<double>();

        this->env_config_.ROBOT_LEG_CONFIG = cfg["robot"]["leg_config"].template As<std::string>();

        //RSINFO("Robot Leg config: " << this->env_config_.ROBOT_LEG_CONFIG);

        this->spinning_ = cfg["control"]["spinning"].template As<bool>();
        this->command_mode_ = (command_t)cfg["control"]["command_mode"].template As<int>();
        this->change_facing_ = (command_t)cfg["control"]["change_facing"].template As<bool>();

        // Create world
        this->world_ = std::make_unique<raisim::World>();
        this->rewards_.initializeFromConfigurationFile(cfg["train"]["reward"]["details"]);

        // Create default terrain
        this->generator_ = WorldGenerator(this->world_.get(), this->anymal_);
        this->hills(0.0, 0.0, 0.0);
        
        std::string urdf_path = cfg["robot"]["urdf_path"].template As<std::string>();
        std::string urdf_base_dir = cfg["robot"]["resource_dir"].template As<std::string>();
        // Add robot
        this->anymal_ = this->world_->addArticulatedSystem(
            resource_dir + urdf_path,
            resource_dir + urdf_base_dir,
            {},
            raisim::COLLISION(2), // Collision group
            -1);
               
        this->anymal_->setName("GiaDoG");
        this->anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        // Get the initial_configuration into a yaml node variable (for convenience)
        Yaml::Node initial_configuration = cfg["robot"]["initial_configuration"];

        std::vector<double> lf_leg_init_angles = {
            initial_configuration["front_left"]["hip"].template As<double>()
            , initial_configuration["front_left"]["thigh"].template As<double>()
            , initial_configuration["front_left"]["shank"].template As<double>()
        };

        std::vector<double> rf_leg_init_angles = {
            initial_configuration["front_right"]["hip"].template As<double>()
            , initial_configuration["front_right"]["thigh"].template As<double>()
            , initial_configuration["front_right"]["shank"].template As<double>()
        };

        std::vector<double> bl_leg_init_angles = {
            initial_configuration["back_left"]["hip"].template As<double>()
            , initial_configuration["back_left"]["thigh"].template As<double>()
            , initial_configuration["back_left"]["shank"].template As<double>()
        };

        std::vector<double> br_leg_init_angles = {
            initial_configuration["back_right"]["hip"].template As<double>()
            , initial_configuration["back_right"]["thigh"].template As<double>()
            , initial_configuration["back_right"]["shank"].template As<double>()
        };

        // Initialization
        this->generalized_coord_dim_ = static_cast<unsigned int>(this->anymal_->getGeneralizedCoordinateDim());
        this->generalized_coord_.setZero(this->generalized_coord_dim_);
        this->generalized_coord_init_.setZero(this->generalized_coord_dim_);
        this->generalized_coord_init_ << 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 
            lf_leg_init_angles[0], lf_leg_init_angles[1], lf_leg_init_angles[2],
            rf_leg_init_angles[0], rf_leg_init_angles[1], rf_leg_init_angles[2],
            bl_leg_init_angles[0], bl_leg_init_angles[1], bl_leg_init_angles[2],
            br_leg_init_angles[0], br_leg_init_angles[1], br_leg_init_angles[2];

        std::pair<double, double> curriculum_params = find_begin_and_decay(
            curriculum_grows_start_,
            curriculum_grows_duration_);
        this->curriculum_base_ = curriculum_params.first;
        this->curriculum_decay_ = curriculum_params.second;
        this->curriculum_coeff_ = pow(this->curriculum_base_, pow(this->curriculum_decay_, this->epoch_));

        this->pos_target_.setZero(this->generalized_coord_dim_);
        this->generalized_vel_dim_ = static_cast<unsigned int>(this->anymal_->getDOF());
        this->n_joints_ = generalized_vel_dim_ - 6;
        this->anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(this->generalized_vel_dim_));
        this->generalized_vel_.setZero(this->generalized_vel_dim_);
        this->generalized_vel_init_.setZero(this->generalized_vel_dim_);
        this->vel_target_.setZero(this->generalized_vel_dim_);
        this->pd_constants_.setZero(this->n_joints_ * 2);

        this->action_dim_ = 16;
        this->action_mean_.setZero(this->action_dim_);
        this->action_scale_.setZero(this->action_dim_);
        Yaml::Node action_mean = cfg["control"]["action"]["mean"];
        Yaml::Node action_scale = cfg["control"]["action"]["scale"];
        for (int i = 0; i < this->action_dim_ - 4; i++)
        {
            if (i % 3 == 0)
            {
                this->action_scale_[i] = action_scale["x_residual"].template As<double>();
                this->action_mean_[i] = action_mean["x_residual"].template As<double>();
            }
            else if (i % 3 == 1)
            {
                this->action_scale_[i] = action_scale["y_residual"].template As<double>();
                this->action_mean_[i] = action_mean["y_residual"].template As<double>();
            }
            else if (i % 3 == 2)
            {
                this->action_scale_[i] = action_scale["z_residual"].template As<double>();
                this->action_mean_[i] = action_mean["z_residual"].template As<double>();
            }
        }
        for (int i = this->action_dim_ - 4; i < this->action_dim_; i++)
        {
            this->action_scale_[i] = action_scale["frequency"].template As<double>();
            this->action_mean_[i] = action_mean["frequency"].template As<double>();
        }

        std::vector<std::string> hip_names = {
            cfg["robot"]["link_names"]["hip_names"]["front_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["hip_names"]["front_right"].template As<std::string>(),
            cfg["robot"]["link_names"]["hip_names"]["back_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["hip_names"]["back_right"].template As<std::string>()
        };
        

        std::vector<std::string> thigh_names = {
            cfg["robot"]["link_names"]["thigh_names"]["front_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["thigh_names"]["front_right"].template As<std::string>(),
            cfg["robot"]["link_names"]["thigh_names"]["back_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["thigh_names"]["back_right"].template As<std::string>()

        };

        std::vector<std::string> shank_names = {
            cfg["robot"]["link_names"]["shank_names"]["front_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["shank_names"]["front_right"].template As<std::string>(),
            cfg["robot"]["link_names"]["shank_names"]["back_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["shank_names"]["back_right"].template As<std::string>()
        };

        std::vector<std::string> foot_names = {
            cfg["robot"]["link_names"]["foot_names"]["front_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["foot_names"]["front_right"].template As<std::string>(),
            cfg["robot"]["link_names"]["foot_names"]["back_left"].template As<std::string>(),
            cfg["robot"]["link_names"]["foot_names"]["back_right"].template As<std::string>()
        };
    
        // Indices of the feet
        for (std::string name : shank_names)
        {
            this->foot_indexes_.insert(this->anymal_->getBodyIdx(name));
        }

        //RSINFO("Initializing contact solver");

        this->contact_solver_ = ContactSolver(
            this->world_.get(),
            this->anymal_,
            this->world_->getTimeStep(),
            1.0, // Fricction coefficient mean
            0.2, // Fricction coefficient std
            hip_names,
            thigh_names,
            shank_names,
            foot_names
            );
        
        //RSINFO("Contact solver set up successfully");
        

        this->base_euler_.setZero(3);
        this->FTG_phases_.setZero(4);
        this->FTG_sin_phases_.setZero(4);
        this->FTG_cos_phases_.setZero(4);
        this->joint_target_.setZero(12);
        this->FTG_frequencies_.setZero(4);
        this->joint_position_.setZero(12);
        this->joint_velocity_.setZero(12);
        this->joint_vel_hist_.setZero(24);
        this->feet_target_pos_.setZero(12);
        this->feet_target_hist_.setZero(24);
        this->joint_pos_err_hist_.setZero(24);

        //RSINFO("Setting up height scanner");


        std::vector<std::string> feet_parent_joints = {
            cfg["robot"]["foot_parent_joints"]["front_left"].template As<std::string>(),
            cfg["robot"]["foot_parent_joints"]["front_right"].template As<std::string>(),
            cfg["robot"]["foot_parent_joints"]["back_left"].template As<std::string>(),
            cfg["robot"]["foot_parent_joints"]["back_right"].template As<std::string>()
        };
        this->height_scanner_ = HeightScanner(
            this->world_.get(),
            this->anymal_,
            &this->env_config_,
            feet_parent_joints,
            cfg["simulation"]["height_scan"]["render"].template As<bool>());
        
        //RSINFO("Height scanner set up successfully");

        // visualize if it is the first environment
        
        if (this->visualizable_)
        {   
            //RSINFO("Setting up visualization components");
            this->server_ = std::make_unique<raisim::RaisimServer>(this->world_.get());
            this->server_->launchServer(port);
            if (this->command_mode_ == command_t::STRAIGHT)
            {
                if (this->display_target_)
                {
                    this->visual_target_ = this->server_->addVisualCylinder(
                        "goal", 0.2, 7, 0, 1, 0, 0.5);
                }
                if (this->display_direction_)
                {
                    this->direction_body_ = this->server_->addVisualPolyLine(
                        "direction_body");
                    this->direction_head_ = this->server_->addVisualSphere(
                        "direction_head", 0.02, 0, 1, 0, 1);
                    this->direction_body_->setColor(0, 1, 0, 1);
                    this->direction_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->direction_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_turning_)
                {
                    this->turning_body_ = this->server_->addVisualPolyLine(
                        "turning_body");
                    this->turning_head_ = this->server_->addVisualSphere(
                        "turning_head", 0.02, 1, 0, 0, 1);
                    this->turning_body_->setColor(1, 0, 0, 1);
                }
                if (this->display_height_)
                {
                    this->height_line_ = this->server_->addVisualPolyLine(
                        "height_line");
                    this->height_line_->setColor(1, 1, 1, 1);
                    this->height_line_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->height_line_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_x_component_)
                {
                    this->x_component_body_ = this->server_->addVisualPolyLine(
                        "x_component_body");
                    this->x_component_head_ = this->server_->addVisualSphere(
                        "x_component_head", 0.02, 1, 0, 0, 1);
                    this->x_component_body_->setColor(1, 0, 0, 1);
                    this->x_component_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->x_component_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_y_component_)
                {
                    this->y_component_body_ = this->server_->addVisualPolyLine(
                        "y_component_body");
                    this->y_component_head_ = this->server_->addVisualSphere(
                        "y_component_head", 0.02, 0, 1, 0, 1);
                    this->y_component_body_->setColor(0, 1, 0, 1);
                    this->y_component_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->y_component_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_z_component_)
                {
                    this->z_component_body_ = this->server_->addVisualPolyLine(
                        "z_component_body");
                    this->z_component_head_ = this->server_->addVisualSphere(
                        "z_component_head", 0.02, 0, 0, 1, 1);
                    this->z_component_body_->setColor(0, 0, 1, 1);
                    this->z_component_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->z_component_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_linear_vel_)
                {
                    this->linear_vel_body_ = this->server_->addVisualPolyLine(
                        "linear_vel_body");
                    this->linear_vel_head_ = this->server_->addVisualSphere(
                        "linear_vel_head", 0.02, 0, 1, 0, 1);
                    this->linear_vel_body_->setColor(0, 1, 0, 1);
                    this->linear_vel_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->linear_vel_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
                if (this->display_angular_vel_)
                {
                    this->angular_vel_body_ = this->server_->addVisualPolyLine(
                        "angular_vel_body");
                    this->angular_vel_head_ = this->server_->addVisualSphere(
                        "angular_vel_head", 0.02, 1, 0, 0, 1);
                    this->angular_vel_body_->setColor(1, 0, 0, 1);
                    this->angular_vel_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->angular_vel_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
            }
            this->height_scanner_.add_visual_indicators(this->server_.get());
            this->server_->focusOn(this->anymal_);
            //RSINFO("Simulation server and visualizers set up successfully.");
        }

        // Initiate the random seed
        srand(uint32_t (time(0)));

        this->latency_ = cfg["simulation"]["latency"]["peak"].template As<double>();
        this->env_config_.CONTROL_DT = 1 / this->latency_;
        double min_latency = cfg["simulation"]["latency"]["min"].template As<double>();
        double max_latency = cfg["simulation"]["latency"]["max"].template As<double>();
        std::array<double, 3> i{min_latency, this->latency_, max_latency};
        std::array<double, 3> w{0, 1, 0};
        this->latency_distribution_ = std::piecewise_linear_distribution<double>{
            i.begin(),
            i.end(),
            w.begin()};

        int sim_steps;
        if (!this->variable_latency_)
        {
            this->control_dt_ = 1 / this->latency_;
            this->env_config_.CONTROL_DT = this->control_dt_;
            sim_steps = int(control_dt_ / simulation_dt_ + 1e-10);
            double dt = this->control_dt_ / sim_steps;

            this->world_->setTimeStep(dt);
        }
        //RSINFO("Simulation time step set to " << this->world_->getTimeStep() << " s");
        //RSINFO("Resetting simulation world");
        this->reset(0);

        //RSINFO("Environment initialized");
    }

    step_t ENVIRONMENT::reset(int epoch)
    {
        this->epoch_ = epoch;
        this->curriculum_coeff_ = pow(this->curriculum_base_, pow(this->curriculum_decay_, this->epoch_));

        // We perform a raycast to get the height of the ground around
        // the x = 0, y = 0. This is used to set the initial height of the
        // robot. This is important because the robot is not standing on
        // the ground when it is created.
        this->generalized_coord_init_[2] = this->env_config_.LEG_SPAN * 1.023 + this->get_terrain_height(0, 0);
        this->anymal_->setState(
            this->generalized_coord_init_,
            this->generalized_vel_init_);

        this->elapsed_time_ = 0.0;
        this->elapsed_steps_ = 0;
        this->traversability_ = 0.0;

        // Re initialize external force applier
        this->external_force_applier_ = ExternalForceApplier(
            this->anymal_,
            this->env_config_.EXTERNAL_FORCE_TIME, // Aplied force duration [Seconds]
            this->env_config_.MAX_EXTERNAL_FORCE   // Max force [Newtons]
        );

        this->set_pd_gains();
        this->change_target();
        this->update_observations();
        this->update_info();

        return {this->observations_, this->rewards_.sum(), false, this->info_};
    }

    step_t ENVIRONMENT::step(const Eigen::Ref<EigenVec> &action)
    {
        // Action scaling
        Eigen::VectorXd action_scaled = action.cast<double>();
        action_scaled = action_scaled.cwiseProduct(this->action_scale_) +
                        this->action_mean_;
        this->update_target();

        // Run FTG
        std::tuple<Eigen::VectorXd,
                   Eigen::VectorXd,
                   Eigen::Vector4d,
                   Eigen::Vector4d,
                   Eigen::Vector4d,
                   Eigen::Vector4d>
            control_pipeline_output;
        control_pipeline_output = control_pipeline(
            action_scaled,
            this->turning_direction_,
            this->target_angle_,
            this->base_euler_(0),
            this->base_euler_(1),
            this->elapsed_time_,
            &this->env_config_);
        this->joint_target_ = std::get<0>(control_pipeline_output);
        this->feet_target_pos_ = std::get<1>(control_pipeline_output);
        this->FTG_frequencies_ = std::get<2>(control_pipeline_output);
        this->FTG_sin_phases_ = std::get<3>(control_pipeline_output);
        this->FTG_cos_phases_ = std::get<4>(control_pipeline_output);
        this->FTG_phases_ = std::get<5>(control_pipeline_output);

        // Update the historical data
        this->joint_pos_err_hist_.tail(12) = this->joint_pos_err_hist_.head(12);
        this->joint_pos_err_hist_.head(12) = this->observations_["joint_position"] - this->joint_target_;

        this->joint_vel_hist_.tail(12) = this->joint_vel_hist_.head(12);
        this->joint_vel_hist_.head(12) = this->observations_["joint_velocity"];

        this->feet_target_hist_.tail(12) = this->feet_target_hist_.head(12);
        this->feet_target_hist_.head(12) = this->feet_target_pos_;

        // Set action
        this->pos_target_.tail(this->n_joints_) = this->joint_target_;
        this->anymal_->setPdTarget(this->pos_target_, this->vel_target_);

        Eigen::VectorXd joint_vel, joint_vel_prev;
        joint_vel.setZero(this->n_joints_);
        joint_vel_prev.setZero(this->n_joints_);
        joint_vel_prev = this->generalized_vel_.tail(12);

        // Advance time in the simulation
        int sim_steps;
        if (this->variable_latency_)
        {
            double latency = this->latency_distribution_(this->random_gen_);
            this->control_dt_ = 1 / latency;
            this->env_config_.CONTROL_DT = this->control_dt_;
            sim_steps = int(control_dt_ / simulation_dt_ + 1e-10);
            double dt = this->control_dt_ / sim_steps;
            this->world_->setTimeStep(dt);
        }
        else
        {
            sim_steps = int(this->control_dt_ / simulation_dt_ + 1e-10);
        }
        for (int i = 0; i < sim_steps; i++)
        {
            this->external_force_applier_.apply_external_force(this->elapsed_time_);
            if (this->server_)
                this->server_->lockVisualizationServerMutex();
            this->world_->integrate();
            if (this->server_)
                this->server_->unlockVisualizationServerMutex();
        }

        // Update acceleration
        this->anymal_->getState(this->generalized_coord_, this->generalized_vel_);
        joint_vel = this->generalized_vel_.tail(12);
        this->joint_acceleration_ = (joint_vel - joint_vel_prev) / this->control_dt_;

        // Traversability calculation
        int traversability = (this->linear_vel_[0] * this->target_direction_[0] +
                             this->linear_vel_[1] * this->target_direction_[1]) >= this->traversability_min_speed_treshold_;
        this->traversability_ = (this->elapsed_steps_ * this->traversability_ +
                                traversability) /
                               (this->elapsed_steps_ + 1);

        // Step the time
        this->elapsed_steps_ += 1;
        this->elapsed_time_ += this->control_dt_;

        // Check if the robot is in the goal
        if (this->current_command_mode_ == command_t::STRAIGHT &&
            (this->target_position_ - this->generalized_coord_.head(2)).norm() < GOAL_RADIUS)
        {
            this->change_target();
        }

        this->update_observations();
        this->register_rewards();
        this->update_info();

        return {this->observations_, this->rewards_.sum(), this->is_terminal_state(), this->info_};
    }

    void ENVIRONMENT::hills(double frequency, double amplitude, double roughness)
    {
        this->generator_.clear();
        this->generator_.hills(frequency, amplitude, roughness);
        this->terrain_ = terrain_t::HILLS;
    }

    void ENVIRONMENT::stairs(double width, double height)
    {
        this->generator_.clear();
        this->generator_.stairs(width, height);
        this->terrain_ = terrain_t::STAIRS;
    }

    void ENVIRONMENT::cellular_steps(double frequency, double amplitude)
    {
        this->generator_.clear();
        this->generator_.cellular_steps(frequency, amplitude);
        this->terrain_ = terrain_t::CELULAR_STEPS;
    }

    void ENVIRONMENT::steps(double widht, double height)
    {
        this->generator_.clear();
        this->generator_.steps(widht, height);
        this->terrain_ = terrain_t::STEPS;
    }

    void ENVIRONMENT::slope(double slope, double roughness)
    {
        this->generator_.clear();
        this->generator_.slope(slope, roughness);
        this->terrain_ = terrain_t::SLOPE;
    }

    void ENVIRONMENT::set_command(double target_angle, int turning_direction, bool stop)
    {
        this->turning_direction_ = turning_direction;
        this->env_config_.CARTESIAN_DELTA = true;

        if (stop)
        {
            this->env_config_.CARTESIAN_DELTA = false;
            this->target_angle_ = 0;
            this->facing_angle_ = 0;
            this->target_position_.setZero(2);
            this->target_direction_.setZero(2);
        }
        else
        {
            this->target_angle_ = target_angle;
            this->target_direction_ << std::cos(target_angle_), std::sin(target_angle_);
        }

        raisim::Vec<4> quat;
        raisim::Mat<3, 3> rot;
        quat[0] = this->generalized_coord_[3];
        quat[1] = this->generalized_coord_[4];
        quat[2] = this->generalized_coord_[5];
        quat[3] = this->generalized_coord_[6];

        raisim::quatToRotMat(quat, rot);

        this->target_position_ = rot.e().block(0, 0, 2, 2).transpose() * this->target_direction_;
    }

    std::map<std::string, int> ENVIRONMENT::get_observations_dimension(void)
    {
        std::map<std::string, int> dimensions;
        for (const std::pair<const std::string, Eigen::VectorXd> &pair : this->observations_)
        {
            dimensions[pair.first] = int (pair.second.size());
        }

        return dimensions;
    }

    int ENVIRONMENT::get_action_dimension(void)
    {
        return 16;
    }

    // TEST METHODS

    void ENVIRONMENT::set_absolute_position(
        double x,
        double y,
        double z,
        double pitch,
        double yaw,
        double roll)
    {
        this->generalized_coord_init_[0] = !std::isnan(x) ? x : this->generalized_coord_[0];
        this->generalized_coord_init_[1] = !std::isnan(y) ? y : this->generalized_coord_[1];
        this->generalized_coord_init_[2] = !std::isnan(z) ? z : this->generalized_coord_[2];

        // We obtain the Euler angles of the current state
        glm::quat q(
            this->generalized_coord_[3],
            this->generalized_coord_[4],
            this->generalized_coord_[5],
            this->generalized_coord_[6]
        );
        glm::vec3 euler_angles = glm::eulerAngles(q);
        double real_pitch = !std::isnan(pitch) ? glm::radians(pitch) : euler_angles.x;
        double real_yaw = !std::isnan(yaw) ? glm::radians(yaw) : euler_angles.y;
        double real_roll = !std::isnan(roll) ? glm::radians(roll) : euler_angles.z;

        // Create the quaternion
        glm::quat real_q = glm::quat(glm::vec3(real_pitch, real_yaw, real_roll));
        this->generalized_coord_init_[3] = real_q.w;
        this->generalized_coord_init_[4] = real_q.x;
        this->generalized_coord_init_[5] = real_q.y;
        this->generalized_coord_init_[6] = real_q.z;


        this->anymal_->setState(
            this->generalized_coord_init_,
            this->generalized_vel_init_);
    }

    void ENVIRONMENT::set_absolute_velocity(
        double linear_x,
        double linear_y,
        double linear_z,
        double angular_x,
        double angular_y,
        double angular_z)
    {
        this->generalized_vel_init_[0] = !std::isnan(linear_x) ? linear_x : this->generalized_vel_[0];
        this->generalized_vel_init_[1] = !std::isnan(linear_y) ? linear_y : this->generalized_vel_[1];
        this->generalized_vel_init_[2] = !std::isnan(linear_z) ? linear_z : this->generalized_vel_[2];
        this->generalized_vel_init_[3] = !std::isnan(angular_x) ? angular_x : this->generalized_vel_[3];
        this->generalized_vel_init_[4] = !std::isnan(angular_y) ? angular_y : this->generalized_vel_[4];
        this->generalized_vel_init_[5] = !std::isnan(angular_z) ? angular_z : this->generalized_vel_[5];
        this->anymal_->setState(
            this->generalized_coord_init_,
            this->generalized_vel_init_);
    }

    // PRIVATE METHODS

    double ENVIRONMENT::get_terrain_height(double x, double y)
    {
        const raisim::RayCollisionList &collision = this->world_->rayTest(
            {x, y, 100},
            {x, y, -100},
            200.,
            true);
        if (collision.size() > 0)
        {
            return collision[0].getPosition()[2];
        }
        else // This should never happen but just in case so the compiler doesn't complain
        {
            return nan("");
        }
    }

    void ENVIRONMENT::set_pd_gains(void)
    {
        Eigen::VectorXd p_vec = Eigen::VectorXd::NullaryExpr(
            this->n_joints_,
            [&]()
            {
                double x = zero_one_real_dist_(this->merssen_twister_);
                return this->p_min_ + x * (this->p_max_ - this->p_min_);
            });

        Eigen::VectorXd d_vec = Eigen::VectorXd::NullaryExpr(
            this->n_joints_,
            [&]()
            {
                double x = zero_one_real_dist_(this->merssen_twister_);
                return this->d_min_ + x * (this->d_max_ - this->d_min_);
            });

        Eigen::VectorXd joint_p_gain(this->generalized_vel_dim_);
        Eigen::VectorXd joint_d_gain(this->generalized_vel_dim_);

        joint_p_gain.setZero();
        joint_d_gain.setZero();
        joint_p_gain.tail(this->n_joints_) << p_vec;
        joint_d_gain.tail(this->n_joints_) << d_vec;

        this->pd_constants_.head(this->n_joints_) << joint_p_gain.tail(this->n_joints_);
        this->pd_constants_.tail(this->n_joints_) << joint_d_gain.tail(this->n_joints_);
        this->anymal_->setPdGains(joint_p_gain, joint_d_gain);
    }

    void ENVIRONMENT::change_target(void)
    {
        if (this->change_facing_)
        {
            this->facing_angle_ = POSIBLE_FACING_ANGLES[rand() % POSIBLE_FACING_ANGLES.size()];
        }

        if (this->spinning_)
        {
            this->turning_direction_ = this->minus_one_one_dist(this->merssen_twister_);
        }

        if (this->command_mode_ == command_t::RANDOM)
        {
            this->current_command_mode_ = (command_t)(rand() % 3);
        }
        else
        {
            this->current_command_mode_ = this->command_mode_;
        }
        double x_size;
        double y_size;
        switch (this->current_command_mode_)
        {
        case command_t::STRAIGHT:
            this->env_config_.CARTESIAN_DELTA = true;
            this->target_angle_ = -M_PI + rand() * (M_PI + M_PI) / RAND_MAX;
            this->target_position_[0] = this->generalized_coord_[0] +
                                        1.5 * std::cos(this->target_angle_);
            if (this->terrain_ == terrain_t::STAIRS)
            {
                this->target_position_[1] = rand() % 100 > 50 ? 10.0 : -10.0;
            }
            else
            {
                this->target_position_[1] = this->generalized_coord_[1] +
                                            1.5 * std::sin(this->target_angle_);
            }

            // Saturate the target_position_ to scale of the terrain.
            x_size = this->generator_.terrain_x_size;
            y_size = this->generator_.terrain_y_size;
            if (std::abs(this->target_position_[0]) > 0.9 * x_size / 2)
            {
                this->target_position_[0] = 0.9 * this->target_position_[0] /
                                            std::abs(this->target_position_[0]) *
                                            x_size / 2;
            }
            if (std::abs(this->target_position_[1]) > 0.9 * y_size / 2)
            {
                this->target_position_[1] = 0.9 * this->target_position_[1] /
                                            std::abs(this->target_position_[1]) *
                                            y_size / 2;
            }

            this->target_direction_ = (this->target_position_ -
                                       this->generalized_coord_.head(2))
                                          .normalized();
            break;

        case command_t::STANCE:
            this->env_config_.CARTESIAN_DELTA = false;
            this->target_angle_ = 0;
            this->facing_angle_ = 0;
            this->turning_direction_ = 0;
            this->target_position_.setZero(2);
            this->target_direction_.setZero(2);
            break;

        case command_t::STATIC_SPIN:
            this->env_config_.CARTESIAN_DELTA = false;
            this->target_angle_ = 0;
            this->facing_angle_ = 0;
            this->target_position_.setZero(2);
            this->target_direction_.setZero(2);
            this->turning_direction_ = this->minus_one_one_dist(this->merssen_twister_);
            break;

        default:
            break;
        }

        this->update_target();
    }

    void ENVIRONMENT::update_target(void)
    {
        // If directed_gait_ is true, change the turning_direction_ to reduce
        // the target_angle_ to 0.
        if (this->current_command_mode_ == command_t::STRAIGHT)
        {
            this->target_direction_ = (this->target_position_ - this->generalized_coord_.head(2)).normalized();
            double oldX = this->target_direction_[0];
            double oldY = this->target_direction_[1];

            // Rotate the target direction to the robot base frame.
            double angle = -this->base_euler_[2];
            this->target_direction_[0] = oldX * std::cos(angle) - oldY * std::sin(angle);
            this->target_direction_[1] = oldX * std::sin(angle) + oldY * std::cos(angle);
            this->target_angle_ = std::atan2(
                this->target_direction_[1],
                this->target_direction_[0]);

            if (std::abs(this->target_angle_ - this->facing_angle_) > M_PI / 6)
            {
                this->turning_direction_ = int ((this->target_angle_ - this->facing_angle_) /
                                           std::abs(this->target_angle_ - this->facing_angle_));
            }
            else
            {
                this->turning_direction_ = 0;
            };
        }

        this->update_visual_objects();
    }

    void ENVIRONMENT::update_visual_objects(void)
    {
        if (this->visualizable_)
        {
            if (this->display_target_ && this->command_mode_ == command_t::STRAIGHT)
            {
                double new_x = this->target_position_[0];
                double new_y = this->target_position_[1];
                double target_height = nan("");

                while (std::isnan(target_height))
                {
                    new_x = new_x * 0.95;
                    new_y = new_y * 0.95;
                    target_height = this->get_terrain_height(new_x, new_y);
                }

                this->visual_target_->setPosition(
                    this->target_position_[0],
                    this->target_position_[1],
                    target_height);
            }
            if (this->display_direction_)
            {
                double scale = 0.3;

                this->direction_body_->clearPoints();
                this->direction_body_->addPoint(
                    this->generalized_coord_.head(3) + Eigen::Vector3d(0, 0, 0.15));
                Eigen::Vector2d tar_dir = (this->target_position_ - this->generalized_coord_.head(2))
                                              .normalized();
                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0] + tar_dir[0] * scale,
                    this->generalized_coord_[1] + tar_dir[1] * scale,
                    this->generalized_coord_[2] + 0.15);
                this->direction_body_->addPoint(direction_head_pos);
                this->direction_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
            if (this->display_turning_)
            {
                this->turning_body_->clearPoints();
                if (this->turning_direction_ != 0)
                {
                    this->turning_body_->addPoint(
                        this->generalized_coord_.head(3) + Eigen::Vector3d(0, 0, 0.15));
                    this->turning_body_->addPoint(
                        this->generalized_coord_.head(3) + Eigen::Vector3d(0, 0, 0.45));

                    double z_pos = (this->turning_direction_ > 0) * 0.3;
                    this->turning_head_->setPosition(
                        this->generalized_coord_.head(3)[0],
                        this->generalized_coord_.head(3)[1],
                        this->generalized_coord_.head(3)[2] + 0.15 + z_pos);
                }
                else
                {
                    this->turning_head_->setPosition(0, 0, 100);
                    this->turning_body_->addPoint(Eigen::Vector3d(0, 0, 200));
                    this->turning_body_->addPoint(Eigen::Vector3d(0, 0, 201));
                }
            }
            if (this->display_height_)
            {
                this->height_line_->clearPoints();
                this->height_line_->addPoint(this->generalized_coord_.head(3));
                this->height_line_->addPoint(
                    this->generalized_coord_.head(3) - Eigen::Vector3d(0, 0, this->body_height_));
            }
            if (this->display_x_component_)
            {
                this->x_component_body_->clearPoints();
                this->x_component_body_->addPoint(
                    this->generalized_coord_.head(3));

                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0] + this->x_component_vector_[0],
                    this->generalized_coord_[1] + this->x_component_vector_[1],
                    this->generalized_coord_[2] + this->x_component_vector_[2]);
                this->x_component_body_->addPoint(direction_head_pos);
                this->x_component_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
            if (this->display_y_component_)
            {
                this->y_component_body_->clearPoints();
                this->y_component_body_->addPoint(
                    this->generalized_coord_.head(3));

                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0] + this->y_component_vector_[0],
                    this->generalized_coord_[1] + this->y_component_vector_[1],
                    this->generalized_coord_[2] + this->y_component_vector_[2]);
                this->y_component_body_->addPoint(direction_head_pos);
                this->y_component_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
            if (this->display_z_component_)
            {
                this->z_component_body_->clearPoints();
                this->z_component_body_->addPoint(
                    this->generalized_coord_.head(3));

                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0] + this->gravity_vector_[0],
                    this->generalized_coord_[1] + this->gravity_vector_[1],
                    this->generalized_coord_[2] + this->gravity_vector_[2]);
                this->z_component_body_->addPoint(direction_head_pos);
                this->z_component_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
            if (this->display_linear_vel_)
            {
                this->linear_vel_body_->clearPoints();
                this->linear_vel_body_->addPoint(
                    this->generalized_coord_.head(3));

                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0] + this->generalized_vel_[0],
                    this->generalized_coord_[1] + this->generalized_vel_[1],
                    this->generalized_coord_[2]);
                this->linear_vel_body_->addPoint(direction_head_pos);
                this->linear_vel_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
            if (this->display_angular_vel_)
            {
                this->angular_vel_body_->clearPoints();
                this->angular_vel_body_->addPoint(
                    this->generalized_coord_.head(3));

                Eigen::Vector3d direction_head_pos(
                    this->generalized_coord_[0],
                    this->generalized_coord_[1],
                    this->generalized_coord_[2] + this->generalized_vel_[5]);
                this->angular_vel_body_->addPoint(direction_head_pos);
                this->angular_vel_head_->setPosition(
                    direction_head_pos[0],
                    direction_head_pos[1],
                    direction_head_pos[2]);
            }
        };
    }

    void ENVIRONMENT::update_observations(void)
    {
        this->anymal_->getState(
            this->generalized_coord_,
            this->generalized_vel_);
        raisim::Vec<4> quat;
        raisim::Mat<3, 3> rot;
        quat[0] = this->generalized_coord_[3];
        quat[1] = this->generalized_coord_[4];
        quat[2] = this->generalized_coord_[5];
        quat[3] = this->generalized_coord_[6];

        raisim::quatToRotMat(quat, rot);
        this->linear_vel_ = rot.e().transpose() * this->generalized_vel_.segment(0, 3);
        this->angular_vel_ = rot.e().transpose() * this->generalized_vel_.segment(3, 3);

        glm::quat q(
            this->generalized_coord_[3],
            this->generalized_coord_[4],
            this->generalized_coord_[5],
            this->generalized_coord_[6]);

        // This way the noise from the orientation is propagated to the
        // gravity vector
        glm::mat4 R = glm::toMat3(q);
        glm::vec3 x_vector = glm::vec3(R[0]);
        glm::vec3 y_vector = glm::vec3(R[1]);
        glm::vec3 gravity = glm::vec3(R[2]);
        this->x_component_vector_ << x_vector.x, x_vector.y, x_vector.z;
        this->y_component_vector_ << y_vector.x, y_vector.y, y_vector.z;
        this->gravity_vector_ << -gravity.x, -gravity.y, -gravity.z;

        /// Body orientation in euler angles
        glm::vec3 euler = glm::eulerAngles(q);
        this->base_euler_ << euler.z + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
            euler.x + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
            euler.y + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_;
        this->height_scanner_.foot_scan(euler.y);

        // Getting body height
        const raisim::RayCollisionList &height_rt = world_->rayTest(
            {generalized_coord_[0], generalized_coord_[1], generalized_coord_[2]},
            {generalized_coord_[0], generalized_coord_[1], generalized_coord_[2] - 50},
            50.,
            true,
            0,
            0,
            raisim::RAISIM_STATIC_COLLISION_GROUP);
        this->body_height_ = (generalized_coord_.head(3) - height_rt[0].getPosition()).norm();


        this->contact_solver_.contact_info();

        this->joint_position_ = this->generalized_coord_.tail(12);
        this->joint_velocity_ = this->generalized_vel_.tail(12);

        this->external_force_applier_.external_force_in_base(rot.e());

        this->observations_["target_direction"] = this->target_direction_;
        this->observations_["turning_direction"][0] = this->turning_direction_;
        this->observations_["body_height"][0] = this->body_height_;
        this->observations_["gravity_vector"] = this->gravity_vector_;
        this->observations_["linear_velocity"] = this->linear_vel_;
        this->observations_["angular_velocity"] = this->angular_vel_;
        this->observations_["joint_position"] = this->joint_position_;
        this->observations_["joint_velocity"] = this->joint_velocity_;
        this->observations_["FTG_sin_phases"] = this->FTG_sin_phases_;
        this->observations_["FTG_cos_phases"] = this->FTG_cos_phases_;
        this->observations_["FTG_frequencies"] = this->FTG_frequencies_;
        this->observations_["base_frequency"][0] = this->env_config_.BASE_FREQUENCY;
        this->observations_["joint_pos_err_hist"] = this->joint_pos_err_hist_;
        this->observations_["joint_vel_hist"] = this->joint_vel_hist_;
        this->observations_["feet_target_hist"] - this->feet_target_hist_;
        this->observations_["terrain_normal"] = this->contact_solver_.terrain_normal;
        this->observations_["feet_height_scan"] = this->height_scanner_.feet_height_scan;
        this->observations_["foot_contact_forces"] = this->contact_solver_.foot_contact_forces;
        this->observations_["foot_contact_states"] = this->contact_solver_.foot_contact_states;
        this->observations_["shank_contact_states"] = this->contact_solver_.shank_contact_states;
        this->observations_["thigh_contact_states"] = this->contact_solver_.thigh_contact_states;
        this->observations_["foot_ground_fricction"] = this->contact_solver_.foot_ground_friction;
        this->observations_["external_force"] = this->external_force_applier_.external_force_base_frame;
        this->observations_["pd_constants"] = this->pd_constants_;

        // Noise addition
        if (this->noise_)
        {
            double noise;
            for (std::pair<std::string, double> param : this->observations_noise_)
            {
                for (int i = 0; i < this->observations_[param.first].size(); i++)
                {
                    noise = param.second * this->norm_dist_(this->random_gen_);
                    this->observations_[param.first](i) += noise;
                }
            }
        }
    }

    void ENVIRONMENT::update_info(void)
    {   
        this->info_["traversability"] = this->traversability_;
        this->info_["projected_speed"] = this->target_direction_.dot(this->linear_vel_.head(2));

        double g = this->world_->getGravity().e().norm();
        double v2 = this->linear_vel_.head(2).squaredNorm();
        this->info_["froude"] = v2 / (g * this->body_height_);

        Eigen::VectorXd torque;
        torque = -50 * (this->joint_position_ - this->joint_target_) - 0.2 * this->joint_velocity_;
        torque = torque.cwiseMin(4.0).cwiseMax(-4.0);
        this->info_["max_torque"] = torque.maxCoeff();

        torque = this->anymal_->getGeneralizedForce().e().tail(12);
        this->info_["power"] = torque.dot(this->joint_velocity_);
    }

    void ENVIRONMENT::register_rewards(void)
    {
        double ort_vel, w_2, v_z, tgs_1st, tgs_2nd;
        double proj_angular_vel = this->turning_direction_ * this->angular_vel_[2];
        double proj_linear_vel = this->target_direction_.dot(this->linear_vel_.head(2));
        Eigen::Vector2d h_angular_vel, h_linear_vel;

        // -------------------------------------------------------------------//
        // Torque reward
        // -------------------------------------------------------------------//
        double torque_reward_ = this->anymal_->getGeneralizedForce().e().tail(12).squaredNorm();
        this->rewards_.record("torque", float(this->curriculum_coeff_ * torque_reward_));

        // -------------------------------------------------------------------//
        // Linear Velocity Reward
        // -------------------------------------------------------------------//
        double linear_vel_reward;
        if (this->current_command_mode_ == command_t::STANCE ||
            this->current_command_mode_ == command_t::STATIC_SPIN)
        {
            linear_vel_reward = std::exp(-1.5 * std::pow(this->linear_vel_.norm(), 2));
        }
        else if (proj_linear_vel < this->env_config_.VEL_TH)
        {
            linear_vel_reward = std::exp(
                -1.5 * std::pow(proj_linear_vel - this->env_config_.VEL_TH, 2));
        }
        else
        {
            linear_vel_reward = 1.0;
        };
        this->rewards_.record("linearVel", float (linear_vel_reward));

        // -------------------------------------------------------------------//
        // Angular Velocity Reward:
        //
        // This reward should only be used when the target direction is not 0.
        // -------------------------------------------------------------------//
        double angular_vel_reward;
        if (this->turning_direction_)
        {
            if (proj_angular_vel < this->env_config_.VEL_TH)
            {
                angular_vel_reward = std::exp(
                    -1.5 * std::pow(proj_angular_vel - this->env_config_.VEL_TH, 2));
            }
            else
            {
                angular_vel_reward = 1.0;
            };
        }
        else
        {
            angular_vel_reward = std::exp(-1.5 * std::pow(this->angular_vel_[2], 2));
        };
        this->rewards_.record("angularVel", float(angular_vel_reward));

        // -------------------------------------------------------------------//
        // Base motion reward
        // -------------------------------------------------------------------//
        double base_motion_reward;
        h_linear_vel = this->linear_vel_.head(2);
        ort_vel = (h_linear_vel - this->target_direction_ * proj_linear_vel).norm();
        h_angular_vel = this->angular_vel_.head(2);
        w_2 = h_angular_vel.dot(h_angular_vel);

        base_motion_reward = std::exp(-1.5 * std::pow(ort_vel, 2)) + std::exp(-1.5 * w_2);
        rewards_.record("baseMotion", float(base_motion_reward));

        // -------------------------------------------------------------------//
        // Body motion Reward:
        //
        // Penalizes the body velocity in directions not part of the command
        // -------------------------------------------------------------------//
        double body_motion_reward;
        v_z = this->linear_vel_(2);
        body_motion_reward = std::exp(-1.5 * std::pow(v_z, 2)) + std::exp(-1.5 * w_2);
        rewards_.record("bodyMotion", float(body_motion_reward));

        // -------------------------------------------------------------------//
        // Linear Orthogonal Velocity Reward:
        //
        // Penalizes the velocity orthogonal to the target direction
        // -------------------------------------------------------------------//
        double linear_orthogonal_vel_reward;
        h_linear_vel = this->linear_vel_.head(2);
        ort_vel = (h_linear_vel - this->target_direction_ * proj_linear_vel).norm();

        linear_orthogonal_vel_reward = std::exp(-3 * std::pow(ort_vel, 2));
        rewards_.record("linearOrthogonalVelocity", float(linear_orthogonal_vel_reward));

        // -------------------------------------------------------------------//
        // Body collision reward:
        //
        // Penalizes undesirable collisions between the robot and the
        // environment. Collisions between articulations that are not the robot
        // feet are penalized.
        // -------------------------------------------------------------------//
        double collisions_reward = this->contact_solver_.undesirable_collisions_reward_;
        this->rewards_.record("bodyCollision", float(this->curriculum_coeff_ * collisions_reward));

        // -------------------------------------------------------------------//
        // Foot Clearance reward:
        //
        // When a leg is in swing phase, the robot should lift the corresponding
        // foot higher than the surroundings to avoid collision
        // -------------------------------------------------------------------//
        double foot_clearance_reward = this->height_scanner_.clearance_reward(this->FTG_phases_);
        this->rewards_.record("footClearance", float(foot_clearance_reward));

        // -------------------------------------------------------------------//
        // Target Smoothness reward:
        //
        // The magnitude of the second order finite difference derivatives of
        // the target foot positions are penalized such that the generated foot
        // trajectories become smoother.
        // We also added the magnitude of the first order derivative.
        // -------------------------------------------------------------------//
        double smoothness_reward;
        tgs_1st = (this->feet_target_pos_ - this->feet_target_hist_.head(12)).norm();
        tgs_2nd = (this->feet_target_pos_ - 2 * this->feet_target_hist_.head(12) +
                   this->feet_target_hist_.tail(12))
                      .norm();

        smoothness_reward = -(tgs_1st + tgs_2nd);
        this->rewards_.record("targetSmoothness", float(this->curriculum_coeff_ * smoothness_reward));

        // -------------------------------------------------------------------//
        // Joint Motion Reward:
        //
        // Penalizes joint velocity and acceleration to avoid vibrations:.
        // -------------------------------------------------------------------//
        double joint_motion_reward;
        joint_motion_reward = -(0.01 * this->generalized_vel_.tail(12).squaredNorm() +
                                this->joint_acceleration_.squaredNorm());
        this->rewards_.record("jointMotion", float(this->curriculum_coeff_ * joint_motion_reward));

        // -------------------------------------------------------------------//
        // Slip Reward:
        //
        // Penalizes the foot velocity if the foot is in contact with the
        // ground to reduce slippage.
        // -------------------------------------------------------------------//
        double slip_reward;
        slip_reward = -(this->height_scanner_.feet_speed_squared_.dot(
            this->contact_solver_.foot_contact_states));
        this->rewards_.record("slip", float(this->curriculum_coeff_ * slip_reward));

        // -------------------------------------------------------------------//
        // Terminal Reward:
        //
        // Heavily penalizes the robot if it falls over.
        // -------------------------------------------------------------------//
        this->rewards_.record("terminal", float(-10 * this->is_terminal_state()));
    }

    bool ENVIRONMENT::is_terminal_state(void)
    {
        // Initialize a boolean variable to keep track of whether any foot is
        // in contact with the ground
        bool foot_contact = false;

        // Check if the robot is in a position where it might have fallen
        if (this->gravity_vector_[2] > -0.6)
        {
            // Iterate over all contacts of the robot
            for (raisim::Contact &contact : this->anymal_->getContacts())
            {
                // Check if the contact is not a foot
                if (
                    this->foot_indexes_.find(contact.getlocalBodyIndex()) ==
                    this->foot_indexes_.end())
                {
                    return true;
                }
                else
                {
                    foot_contact = true;
                }
            }

            // If the foot_contact variable is still false, return true to
            // indicate that the robot has fallen because neither foot is in
            // contact
            if (!foot_contact)
            {
                return true;
            }
        }

        return false;
    }

    thread_local std::mt19937 raisim::ENVIRONMENT::random_gen_;
}