#include "Environment.hpp"

namespace raisim
{
    ENVIRONMENT::ENVIRONMENT(
        const std::string &resource_dir,
        const Yaml::Node &cfg,
        bool visualizable,
        std::vector<std::string> non_privileged_obs,
        std::vector<std::string> privileged_obs,
        std::vector<std::string> historic_obs,
        int port) : RaisimGymEnv(resource_dir, cfg,
                                visualizable, 
                                non_privileged_obs,
                                privileged_obs,
                                historic_obs,
                                port),
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
        
        this->case_1_prob_ = cfg["control"]["command_mode_probabilities"]["front_facing_straight"].template As<double>();
        this->case_2_prob_ = cfg["control"]["command_mode_probabilities"]["random_facing_straight"].template As<double>();
        this->case_3_prob_ = cfg["control"]["command_mode_probabilities"]["static_spin"].template As<double>();
        this->case_4_prob_ = cfg["control"]["command_mode_probabilities"]["stance"].template As<double>();
        this->case_5_prob_ = cfg["control"]["command_mode_probabilities"]["fixed_direction"].template As<double>();
        
        // Normalize probabilities
        double sum = this->case_1_prob_ + this->case_2_prob_ + this->case_3_prob_ + this->case_4_prob_ + this->case_5_prob_;
        this->case_1_prob_ /= sum;
        this->case_2_prob_ /= sum;
        this->case_3_prob_ /= sum;
        this->case_4_prob_ /= sum;
        this->case_5_prob_ /= sum;


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

        // Iterate over this node cfg["observations"]["privileged_observations"]
        // And push the keys into the privileged_observations_keys_ vector
        this->regular_observations_keys_ = non_privileged_obs;
        this->privileged_observations_keys_ = privileged_obs;
        this->historic_observations_keys_ =  historic_obs;
        
        // if both are empty, then we use all the observations as regular observations
        if (this->regular_observations_keys_.empty() && this->privileged_observations_keys_.empty())
        {
            for (const auto &pair : this->observations_)
            {
                this->regular_observations_keys_.push_back(pair.first);
            }
        }
        
        // Check that all the historic observations are also regular observations
        for (const std::string &key: this->historic_observations_keys_)
        {   
            auto itr = std::find(this->regular_observations_keys_.begin(), this->regular_observations_keys_.end(), key);
            if (itr == this->regular_observations_keys_.end())
            {
                throw std::runtime_error("Historic observation " + key + " is not a regular observation");
            }
        }
        // Check that all the historic observations are consecutive in the regular observations
        for (int i = 0; i < this->historic_observations_keys_.size() - 1; i++)
        {
            auto itr = std::find(this->regular_observations_keys_.begin(), this->regular_observations_keys_.end(), this->historic_observations_keys_[i]);
            auto itr_next = std::find(this->regular_observations_keys_.begin(), this->regular_observations_keys_.end(), this->historic_observations_keys_[i + 1]);
            if (itr_next - itr != 1)
            {
                throw std::runtime_error("Historic observations are not consecutive in the non priviledge observations");
            }
        }

        // Check that all the privileged observations are not in the regular observations
        for (const std::string &key: this->privileged_observations_keys_)
        {   
            auto itr = std::find(this->regular_observations_keys_.begin(), this->regular_observations_keys_.end(), key);
            if (itr != this->regular_observations_keys_.end())
            {
                throw std::runtime_error("Privileged observation " + key + " is also a regular observation");
            }
        }

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
        // TODO: Change the friction coefficient to be a parameter
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

        this->observations_["FTG_frequencies"] = this->FTG_frequencies_;
        this->observations_["joint_position"] = this->joint_position_;
        this->observations_["joint_velocity"] = this->joint_velocity_;
        

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
            cfg["simulation"]["height_scan"]["render"].template As<bool>()
            && this->visualizable_);
        
        
        // Allocate the memory for the feet height scan as it is the only observation
        // that is not of fixed size at compile time
        this->observations_["feet_height_scan"] = Eigen::VectorXd::Zero(this->height_scanner_.n_scans_); 
        if (this->visualizable_)
        {   
            //RSINFO("Setting up visualization components");
            this->server_ = std::make_unique<raisim::RaisimServer>(this->world_.get());
            this->server_->launchServer(port);
            
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
        
        this->historic_obs_size_ = 0;
        for (const std::string &key : this->historic_observations_keys_){
            this->historic_obs_size_ += this->observations_[key].size();
        }

        // Pre-allocate the memmory for observations vector
        this->regular_obs_size_ = 0;
        for (const std::string &key : this->regular_observations_keys_){
            this->regular_obs_size_ += this->observations_[key].size();
            this->observations_sizes_[key] = this->observations_[key].size();
        }

        this->privileged_obs_size_ = 0;
        for (const std::string &key : this->privileged_observations_keys_){
           this->privileged_obs_size_ += this->observations_[key].size();
           this->observations_sizes_[key] = this->observations_[key].size();
        }

        this->regular_obs_begin_idx_ = 0;
        this->historic_obs_begin_idx_ = 0;
        this->privileged_obs_begin_idx_ = this->regular_obs_size_;
        this->obs_size_ = this->regular_obs_size_ + this->privileged_obs_size_;
        this->observations_vector_ = Eigen::VectorXd::Zero(this->regular_obs_size_  + this->privileged_obs_size_);
        
        
       
        this->reset(0);

    }

    step_t ENVIRONMENT::reset(int epoch)
    {

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
        // REset ok
        return {this->observations_vector_, this->rewards_.sum(), false, this->info_};
    }

    step_t ENVIRONMENT::step(const Eigen::Ref<EigenVec> &action)
    {
        // Action scaling
        Eigen::VectorXd action_scaled = action.cast<double>();
        action_scaled = action_scaled.cwiseProduct(this->action_scale_) +
                        this->action_mean_;
        this->update_target();
        // Print the action
        

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

        // Dessaturate the elapsed time if it is greater than the max time ak.a reset the clock this
        // is to avoid numerical errors in the FTG
        if (this->elapsed_time_ > 5.0)
        {
            this->elapsed_time_ = 0;
        }

        // Check if the robot is in the goal
        if (this->current_command_mode_ == command_t::STRAIGHT &&
            (this->target_position_ - this->generalized_coord_.head(2)).norm() < GOAL_RADIUS)
        {   
            this->change_target(true);
        }
        this->update_observations();
        this->register_rewards();
        this->update_info();


        return {this->observations_vector_, this->rewards_.sum(), this->is_terminal_state(), this->info_};
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
        this->env_config_.ANGULAR_DELTA = true;

        this->current_command_mode_ == command_t::EXTERNAL;

        if (stop)
        {
            this->env_config_.CARTESIAN_DELTA = false;
            this->env_config_.ANGULAR_DELTA = false;
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

        Eigen::Matrix2d rot_eigen;

        rot_eigen << rot(0, 0), rot(0, 1),rot(1, 0), rot(1, 1);

        Eigen::Vector2d vis_direction;

        vis_direction << std::cos(target_angle_ ), std::sin(target_angle_ );

        vis_direction = rot_eigen * vis_direction;

        // Rotate the direction vector
        this->target_position_ = {
            this->generalized_coord_[0] + vis_direction[0],
            this->generalized_coord_[1] + vis_direction[1]
            };
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

    void ENVIRONMENT::update_curriculum_coefficient(void){
        this->epoch_ += 1;
        this->curriculum_coeff_ = pow(this->curriculum_base_, pow(this->curriculum_decay_, this->epoch_));
    }

    void ENVIRONMENT::set_curriculum_coefficient(double value){
        if (value>1){value = 1;};
        if (value<0){value = 0;};
        this->curriculum_coeff_ = value;
    }

    // TEST METHODS

    void ENVIRONMENT::set_foot_positions_and_base_pose(const Eigen::Ref<EigenVec> &foot_pos,
        double x,
        double y,
        double z,
        double roll,
        double pitch,
        double yaw
        )
    {
        Eigen::VectorXd foot_pos_vec = foot_pos.cast<double>();

        Eigen::VectorXd target_joint_angles = Eigen::VectorXd::Zero(12);
        for (int i = 0; i < 4; i++)
        {
        double x = foot_pos_vec(i * 3);
        double y = foot_pos_vec(i * 3 + 1) ;
        double z = foot_pos_vec(i * 3 + 2);


        Eigen::Vector3d r;
        if (this->env_config_.USE_HORIZONTAL_FRAME){
            x += 0;
            y += this->env_config_.H_OFF * pow(-1, i);
            z += -this->env_config_.LEG_SPAN * (1 - 0.225); 
        }

        double roll = this->base_euler_(0);
        double pitch = this->base_euler_(1);

        r = {
            x * std::cos(pitch) + y * std::sin(pitch) * std::sin(roll) + z * std::sin(pitch) * std::cos(roll) + 0,
            0 + y * std::cos(roll) - z * std::sin(roll),
            -x * std::sin(pitch) + y * std::cos(pitch) * std::sin(roll) + z * std::cos(pitch) * std::cos(roll)};
        
        if (!this->env_config_.USE_HORIZONTAL_FRAME){
            r(1) += this->env_config_.H_OFF * pow(-1, i);
            r(2) += -this->env_config_.LEG_SPAN * (1 - 0.225); 
        }

        if ( (i == 2 || i == 3) && this->env_config_.ROBOT_LEG_CONFIG == "><"){
            r(0) = -r(0);
        }

        bool right_leg = i == 1 || i == 3;
  
        // Asing the joint angles to the joint angle vector.
        auto leg_joint_angles = solve_leg_IK(right_leg, r, &this->env_config_);

        if ( (i == 2 || i == 3) && this->env_config_.ROBOT_LEG_CONFIG == "><"){
            target_joint_angles(i * 3)     = leg_joint_angles[0];
            target_joint_angles(i * 3 + 1) = -leg_joint_angles[1];
            target_joint_angles(i * 3 + 2) = -leg_joint_angles[2];
        }
        else{  
            target_joint_angles(i * 3)     = leg_joint_angles[0];
            target_joint_angles(i * 3 + 1) = leg_joint_angles[1];
            target_joint_angles(i * 3 + 2) = leg_joint_angles[2];
        }  
        }

        int sim_steps = int(this->control_dt_ / simulation_dt_ + 1e-10);
        this->pos_target_.tail(this->n_joints_) = target_joint_angles;
        this->anymal_->setPdTarget(this->pos_target_, this->vel_target_);
        this->joint_position_ = target_joint_angles;
        for (int i = 0; i < sim_steps; i++)
        {
            if (this->server_)
                this->server_->lockVisualizationServerMutex();
            this->set_absolute_position(x, y, z, pitch, yaw, roll);
            this->world_->integrate();
            if (this->server_)
                this->server_->unlockVisualizationServerMutex();
        }
        
    }

    

    void ENVIRONMENT::set_gait_config(
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
        ){
            this->env_config_.BASE_FREQUENCY  = !std::isnan(base_frequency) ? base_frequency : this->env_config_.BASE_FREQUENCY;

            this->env_config_.SIGMA_0[0] = !std::isnan(leg_1_phase) ? leg_1_phase : this->env_config_.SIGMA_0[0];
            this->env_config_.SIGMA_0[1] = !std::isnan(leg_2_phase) ? leg_2_phase : this->env_config_.SIGMA_0[1];
            this->env_config_.SIGMA_0[2] = !std::isnan(leg_3_phase) ? leg_3_phase : this->env_config_.SIGMA_0[2];
            this->env_config_.SIGMA_0[3] = !std::isnan(leg_4_phase) ? leg_4_phase : this->env_config_.SIGMA_0[3];

            this->env_config_.H = !std::isnan(foot_vertical_span) ? foot_vertical_span : this->env_config_.H;
            
            this->env_config_.X_MOV_DELTA = !std::isnan(x_movement_delta) ? x_movement_delta : this->env_config_.X_MOV_DELTA;
            this->env_config_.Y_MOV_DELTA = !std::isnan(y_movement_delta) ? y_movement_delta : this->env_config_.Y_MOV_DELTA;
            this->env_config_.ANG_MOV_DELTA = !std::isnan(angular_movement_delta) ? angular_movement_delta : this->env_config_.ANG_MOV_DELTA;

            this->env_config_.LEG_SPAN  = !std::isnan(leg_span) ? leg_span : this->env_config_.LEG_SPAN;

            this->env_config_.USE_HORIZONTAL_FRAME = !std::isnan(use_horizontal_frame) ? use_horizontal_frame : this->env_config_.USE_HORIZONTAL_FRAME ;
            

        }

    void ENVIRONMENT::set_absolute_position(
        double x,
        double y,
        double z,
        double roll,
        double pitch,
        double yaw
        )
    {
        this->generalized_coord_init_[0] = !std::isnan(x) ? x : this->generalized_coord_[0];
        this->generalized_coord_init_[1] = !std::isnan(y) ? y : this->generalized_coord_[1];
        this->generalized_coord_init_[2] = !std::isnan(z) ? z : this->generalized_coord_[2];

        Quaternion qt{
            this->generalized_coord_[3],
            this->generalized_coord_[4],
            this->generalized_coord_[5],
            this->generalized_coord_[6]
        };

        EulerAngles euler = to_euler_angles(qt);

        double real_roll = !std::isnan(roll) ? glm::radians(roll) : euler.roll;
        double real_pitch = !std::isnan(pitch) ? glm::radians(pitch) : euler.pitch;
        double real_yaw = !std::isnan(yaw) ? glm::radians(yaw) : euler.yaw;
        
        EulerAngles real_euler{real_roll, real_pitch, real_yaw};
        // Create the quaternion
        Quaternion real_q = to_quaternion(real_euler);
        this->generalized_coord_init_[3] = real_q.w;
        this->generalized_coord_init_[4] = real_q.x;
        this->generalized_coord_init_[5] = real_q.y;
        this->generalized_coord_init_[6] = real_q.z;

        // Get the current velocity from raisim
        this->generalized_coord_init_.tail(12) = this->joint_position_;
        this->generalized_vel_init_.tail(12) = this->joint_velocity_;

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

    void ENVIRONMENT::change_target(bool preserve_command_mode)
    {
        if (this->change_facing_)
        {
            this->facing_angle_ = POSIBLE_FACING_ANGLES[rand() % POSIBLE_FACING_ANGLES.size()];
        }

        if (this->spinning_)
        {
            this->turning_direction_ = this->minus_one_one_dist(this->merssen_twister_);
        }

        if (this->command_mode_ == command_t::RANDOM && !preserve_command_mode)
        {
            this->current_command_mode_ = (command_t)(rand() % 3);
        }
        else if (this->command_mode_ == command_t::PROBABILITY && !preserve_command_mode)
        {
            double prob = zero_one_real_dist_(this->merssen_twister_);

            // Case 1: We want the robot to go straight to the target facing it 
            if (prob < this->case_1_prob_){
                this->current_command_mode_ = command_t::STRAIGHT;
                this->facing_angle_ = 0;
                this->turning_direction_ = 0;
            }
            // Case 2: Same as case 1 but with a random facing angle
            else if (prob < this->case_1_prob_ + this->case_2_prob_)
            {
                this->current_command_mode_ = command_t::STRAIGHT;
                this->facing_angle_ = POSIBLE_FACING_ANGLES[rand() % POSIBLE_FACING_ANGLES.size()];
                this->turning_direction_ = 0;
            }
            // Case 3: Now we want the robot spin standing in place
            else if (prob < this->case_1_prob_ + this->case_2_prob_ + this->case_3_prob_)
            {
                this->current_command_mode_ = command_t::STATIC_SPIN; 
            }
            // Case 4: The robot stays in one place
            else if (prob < this->case_1_prob_ + this->case_2_prob_ + this->case_3_prob_ + this->case_4_prob_)
            {
                this->current_command_mode_ = command_t::STANCE; 
            }  
            // Case 5: The robot moves in a fixed direction
            else
            {
                this->current_command_mode_ = command_t::FIXED_DIRECTION; 
            }
        }
        else if (preserve_command_mode){
            this->current_command_mode_ = this->current_command_mode_;
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
            this->env_config_.ANGULAR_DELTA = true;
            
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
            this->env_config_.ANGULAR_DELTA = true;
            this->target_angle_ = 0;
            this->facing_angle_ = 0;
            this->target_position_.setZero(2);
            this->target_direction_.setZero(2);
            this->turning_direction_ = this->minus_one_one_dist(this->merssen_twister_);
            break;
        
        case command_t::FIXED_DIRECTION:    
            this->turning_direction_ = 0;
            this->target_angle_ = -M_PI + rand() * (M_PI + M_PI) / RAND_MAX;
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
            
            this->env_config_.CARTESIAN_DELTA = true;
            this->env_config_.ANGULAR_DELTA = true;

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
                    this->generalized_coord_[0] + this->z_component_vector_[0],
                    this->generalized_coord_[1] + this->z_component_vector_[1],
                    this->generalized_coord_[2] + this->z_component_vector_[2]);
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
        this->linear_vel_  = rot.e().transpose() * this->generalized_vel_.segment(0, 3);
        this->angular_vel_ = rot.e().transpose() * this->generalized_vel_.segment(3, 3);

        Quaternion qt;

        qt.w = this->generalized_coord_[3];
        qt.x = this->generalized_coord_[4];
        qt.y = this->generalized_coord_[5];
        qt.z = this->generalized_coord_[6];

        EulerAngles euler = to_euler_angles(qt);
        
        this->base_euler_ << euler.roll + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
                             euler.pitch + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_,
                             euler.yaw + this->norm_dist_(this->random_gen_) * this->orientation_noise_std_;
        

        this->gravity_vector_ << - std::sin(base_euler_[1] ),
                std::sin(base_euler_[0]) * std::cos(base_euler_[1]),
                std::cos(base_euler_[0]) * std::cos(base_euler_[1]);
        
        
        // Calculate the rotation matrix with the negative of the euler angles
        // This is done to get the rotation matrix from the world frame to the body frame
        // This is done because the gravity vector is defined in the body frame
        // But for the visualization, we need the gravity vector in the world frame
        Eigen::Matrix3d new_rot;

        euler.roll = -euler.roll;
        euler.pitch = -euler.pitch;
        euler.yaw = -euler.yaw;

        new_rot = euler_to_rotation_matrix(euler);

        this->x_component_vector_ << new_rot.row(0).transpose();
        this->y_component_vector_ << new_rot.row(1).transpose();
        this->z_component_vector_ << new_rot.row(2).transpose();
        
        this->height_scanner_.foot_scan(euler.yaw);

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
        
        this->observations_["target_direction"] << this->target_direction_;
        this->observations_["turning_direction"][0] = this->turning_direction_;
        this->observations_["body_height"][0] = this->body_height_;
        this->observations_["gravity_vector"] << this->gravity_vector_;
        this->observations_["linear_velocity"] << this->linear_vel_;
        this->observations_["angular_velocity"] << this->angular_vel_;
        this->observations_["joint_position"] << this->joint_position_;
        this->observations_["joint_velocity"] << this->joint_velocity_;
        this->observations_["FTG_sin_phases"] << this->FTG_sin_phases_;
        this->observations_["FTG_cos_phases"] << this->FTG_cos_phases_;
        this->observations_["FTG_frequencies"] << this->FTG_frequencies_;
        this->observations_["base_frequency"][0] = this->env_config_.BASE_FREQUENCY;
        this->observations_["joint_pos_err_hist"] << this->joint_pos_err_hist_;
        this->observations_["joint_vel_hist"] << this->joint_vel_hist_;
        this->observations_["feet_target_hist"] << this->feet_target_hist_;
        this->observations_["terrain_normal"] << this->contact_solver_.terrain_normal;
        this->observations_["feet_height_scan"] << this->height_scanner_.feet_height_scan;
        this->observations_["foot_contact_forces"] << this->contact_solver_.foot_contact_forces;
        this->observations_["foot_contact_states"] << this->contact_solver_.foot_contact_states;
        this->observations_["shank_contact_states"] << this->contact_solver_.shank_contact_states;
        this->observations_["thigh_contact_states"] << this->contact_solver_.thigh_contact_states;
        this->observations_["foot_ground_fricction"] << this->contact_solver_.foot_ground_friction;
        this->observations_["external_force"] << this->external_force_applier_.external_force_base_frame;
        this->observations_["pd_constants"] << this->pd_constants_;

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
        // Fill the regular observations vector and the privileged observations vector
        int i = 0;
        int obs_size;
        for (const std::string &key : this->regular_observations_keys_){
            obs_size = this->observations_[key].size();
            this->observations_vector_.segment(i,  i + obs_size) << this->observations_[key];
            i += obs_size;
        }
        // assert that the privileged observations are at the end of the observations vector (Debugging)
        assert(i == this->privileged_obs_begin_idx_);
        i = this->privileged_obs_begin_idx_;
        for (const std::string &key : this->privileged_observations_keys_){
            obs_size = this->observations_[key].size();
            this->observations_vector_.segment(i,  i + obs_size) << this->observations_[key];
            i += obs_size;
        }
    }

    std::map<std::string, std::array<int, 2>> ENVIRONMENT::get_observations_indexes(){
        std::map<std::string, std::array<int, 2>> indexes;
        int i = 0;
        int obs_size;
        std::array<int, 2> idx;
        for (const std::string &key : this->regular_observations_keys_){
            obs_size = this->observations_sizes_[key];
            idx[0] = i;
            idx[1] = i + obs_size;
            indexes[key] = idx;
            i += obs_size;
        }
        // assert that the privileged observations are at the end of the observations vector (Debugging)
        assert(i == this->privileged_obs_begin_idx_);
        i = 0;
        for (const std::string &key : this->privileged_observations_keys_){
            obs_size = this->observations_sizes_[key];
            idx[0] = i;
            idx[1] = i + obs_size;
            indexes[key] = idx;
            i += obs_size;
        }
        return indexes;
    }

    void ENVIRONMENT::update_info(void)
    {   
        this->info_["traversability"] = this->traversability_;
        this->info_["projected_speed"] = this->target_direction_.dot(this->linear_vel_.head(2));

        double g = this->world_->getGravity().e().norm();
        double v2 = this->linear_vel_.head(2).squaredNorm();
        if (this->body_height_ > 0){
            this->info_["froude"] = v2 / (g * this->body_height_);
        }
        else{
            this->info_["froude"] = 0;
        }

        Eigen::VectorXd torque;
        torque = this->anymal_->getGeneralizedForce().e().tail(12);
        this->info_["max_torque"] = torque.maxCoeff();
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
        // Previous version of the reward function (is buggy, might crash!)
        // -------------------------------------------------------------------//
        // double base_motion_reward;
        // h_linear_vel = this->linear_vel_.head(2);
        // ort_vel = (h_linear_vel - this->target_direction_ * proj_linear_vel).norm();
        // h_angular_vel = this->angular_vel_.head(2);
        // w_2 = h_angular_vel.dot(h_angular_vel);

        // base_motion_reward = std::exp(-1.5 * std::pow(ort_vel, 2)) + std::exp(-1.5 * w_2);
        // rewards_.record("baseMotion", float(base_motion_reward));

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
        if (std::isnan(smoothness_reward))
        {
            smoothness_reward = 0.0;
        }
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
        if (this->gravity_vector_[2] < 0.6)
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