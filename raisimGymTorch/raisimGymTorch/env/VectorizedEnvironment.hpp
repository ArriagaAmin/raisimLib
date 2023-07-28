//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "omp.h"
#include <chrono>
#include <thread>
#include <iostream>
#include "Yaml.hpp"
#include "RaisimGymEnv.hpp"

namespace raisim
{
    int THREAD_COUNT;

    /**
     * @brief Class that stores the parameters to normalize the observations
     *
     */
    struct statistics_t
    {
        // Mean of each observation
        Eigen::VectorXd mean;
        // Variance of each observation
        Eigen::VectorXd var;
        // Total samples performed
        float count;
    };

    struct step_array_info_t{
        Eigen::MatrixXd non_privileged_observations;
        Eigen::MatrixXd privileged_observations;
        Eigen::MatrixXd historic_observations;
        Eigen::VectorXd rewards;
        Eigen::VectorXd dones;
        Eigen::VectorXd traversability;
        Eigen::VectorXd froude;
        Eigen::VectorXd projected_speed;
        Eigen::VectorXd max_torque;
        Eigen::VectorXd power;
    };


    template <class ChildEnvironment>
    class VectorizedEnvironment
    {

    private:
        // Individual environments
        std::vector<ChildEnvironment *> environments_;

        // Port through which you can connect to the simulation visualizer.
        int port_;
        // Current epoch
        int epoch_ = 0;
        // Environments configuration
        Yaml::Node cfg_;
        // Number of environments in parallel
        int num_envs_ = 1;
        // Action space dimension
        int action_dim = 0;
        // Indicates if a simulation will be displayed visually
        bool render_ = false;
        // Indicates that the environment will restart automatically if it
        // detects that the robot falls
        bool auto_reset = true;
        // String that contains the environment configuration
        std::string cfg_string_;
        // Directory where the resources needed to build the environment
        // are located
        std::string resource_dir_;
        // Dimension of the space of each observation
        std::map<std::string, int> observation_dimensions_;

        /// Indicates whether the observations should be normalized
        bool normalize_ = true;
        
        // Total samples performed
        float count_ = 1e-4f;

        // We have to get the size of the observation space of each environment
        // to create the observation vector
        int regular_obs_begin_idx_;
        int privileged_obs_begin_idx_;
        int historic_obs_begin_idx_;
        
        int regular_obs_size_;
        int privileged_obs_size_;
        int historic_obs_size_;
        int obs_size_;
        

        Eigen::VectorXd mean_;
        Eigen::VectorXd var_;
        Eigen::VectorXd recent_mean_;
        Eigen::VectorXd recent_var_;
        Eigen::VectorXd delta_;
        Eigen::VectorXd epsilon_;

        Eigen::VectorXd dones_ ;
        Eigen::VectorXd rewards_ ;
        Eigen::VectorXd traversability_ ;
        Eigen::VectorXd froude_ ;
        Eigen::VectorXd projected_speed_ ;
        Eigen::VectorXd max_torque_ ;
        Eigen::VectorXd power_ ;

        Eigen::MatrixXd observations_;

        std::vector<std::string> non_privileged_obs_;
        std::vector<std::string> privileged_obs_;
        std::vector<std::string> historic_obs_; 

       

        void update_statistics(Eigen::MatrixXd &observations, bool update)
        {
            if (update)
            {   
                this->recent_mean_ = observations.colwise().mean();
                this->recent_var_ = (observations.rowwise() - this->recent_mean_.transpose()).colwise().squaredNorm() / this->num_envs_;

                this->delta_ = this->mean_ - this->recent_mean_;
                
                for(int i=0; i < this->obs_size_; i++)
                    this->delta_[i] = this->delta_[i]*this->delta_[i];

                float total_count = this->count_ + num_envs_;

                this->mean_  = this->mean_ * (this->count_ / total_count) + this->recent_mean_ * (num_envs_ / total_count);
                this->var_ = (this->var_ * this->count_ + this->recent_var_ * num_envs_ + this->delta_ * (this->count_ * num_envs_ / total_count)) / (total_count);
                this->count_ = total_count;
            }

#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            // Normalize each vector in each map
            for (int i = 0; i < this->num_envs_; i++)
            {   
                observations.row(i) = (observations.row(i) - this->mean_.transpose()).template cwiseQuotient<>((this->var_ + this->epsilon_).cwiseSqrt().transpose());
            }
        }

    public:
        explicit VectorizedEnvironment(
            std::string resource_dir,
            std::string cfg,
            int port,
            bool normalize,
            std::vector<std::string> non_privileged_obs,
            std::vector<std::string> privileged_obs,
            std::vector<std::string> historic_obs
            ) : resource_dir_(resource_dir),
                cfg_string_(cfg),
                normalize_(normalize),
                port_(port),
                non_privileged_obs_(non_privileged_obs),
                privileged_obs_(privileged_obs),
                historic_obs_(historic_obs)
        {
            Yaml::Parse(this->cfg_, cfg);

            this->auto_reset = this->cfg_["control"]["auto_reset"].template As<bool>();
            if (&this->cfg_["simulation"]["render"])
                this->render_ = this->cfg_["simulation"]["render"].template As<bool>();

            // Print the configuration
            RSINFO("#---------------------------------------------------------#")
            RSINFO("\033[1mINITIALIZING VECTORIZED ENVIRONMENT\033[0m");
            RSINFO("\033[1mConfiguration:\033[0m");
            RSINFO("  - Resource directory: " << this->resource_dir_);
            RSINFO("\033[1mSimulation:\033[0m");
            RSINFO("  - Number of environments: " << this->cfg_["simulation"]["num_envs"].template As<int>());
            RSINFO("  - Number of threads: " << this->cfg_["simulation"]["num_threads"].template As<int>());
            RSINFO("  - Normalize: " << (this->normalize_ ? "true" : "false"));
            RSINFO("  - Render: " << (this->render_ ? "true" : "false"));
            RSINFO("  - Render port: " << this->port_);
            RSINFO("  - Noise: " << (this->cfg_["simulation"]["noise"].template As<bool>() ? "true" : "false"));
            RSINFO("  - Differencial time: " << this->cfg_["simulation"]["simulation_dt"].template As<float>() << "sec");
            if (this->cfg_["simulation"]["latency"]["variable"].template As<bool>())
            {
                float min = this->cfg_["simulation"]["latency"]["min"].template As<float>();
                float peak = this->cfg_["simulation"]["latency"]["peak"].template As<float>();
                float max = this->cfg_["simulation"]["latency"]["max"].template As<float>();
                RSINFO("  - Latency: (" << min << ", " << peak << ", " << max << ") (Variable)");
            }
            else
            {
                RSINFO("  - Latency: " << this->cfg_["simulation"]["latency"]["peak"].template As<float>() << " (Fixed)");
            }
            RSINFO("\033[1mControl:\033[0m");
            RSINFO("  - Command mode: " << this->cfg_["control"]["command_mode"].template As<std::string>());
            RSINFO("  - Spinning: " << (this->cfg_["control"]["spinning"].template As<bool>() ? "true" : "false"));
            RSINFO("  - Change facing: " << (this->cfg_["control"]["change_facing"].template As<bool>() ? "true" : "false"));
            RSINFO("  - Auto-reset: " << (this->auto_reset ? "true" : "false"));
            RSINFO("#---------------------------------------------------------#\n")
            std::this_thread::sleep_for(std::chrono::seconds(2));

            // Initialize the environments
            init();
        }

        ~VectorizedEnvironment()
        {
            for (auto *ptr : this->environments_)
                delete ptr;
        }

        /**
         * @brief Initialize all environments with the indicated configuration
         *
         */
        void init(void)
        {
            THREAD_COUNT = this->cfg_["simulation"]["num_threads"].template As<int>();
            omp_set_num_threads(THREAD_COUNT);
            this->num_envs_ = this->cfg_["simulation"]["num_envs"].template As<int>();

            this->environments_.reserve(this->num_envs_);
            for (int i = 0; i < this->num_envs_; i++)
            {
                this->environments_.push_back(new ChildEnvironment(
                    this->resource_dir_,
                    this->cfg_,
                    this->render_ && i == 0,
                    this->non_privileged_obs_,
                    this->privileged_obs_,
                    this->historic_obs_,
                    this->port_));
            }


            this->observation_dimensions_ = this->environments_[0]->get_observations_dimension();
            this->action_dim = this->environments_[0]->get_action_dimension();

            this->regular_obs_begin_idx_ = this->environments_[0]->regular_obs_begin_idx_;
            this->privileged_obs_begin_idx_= this->environments_[0]->privileged_obs_begin_idx_;
            this->historic_obs_begin_idx_= this->environments_[0]->historic_obs_begin_idx_;
            
            this->regular_obs_size_= this->environments_[0]->regular_obs_size_;
            this->privileged_obs_size_= this->environments_[0]->privileged_obs_size_;
            this->historic_obs_size_= this->environments_[0]->historic_obs_size_;
            this->obs_size_= this->environments_[0]->obs_size_;
            // Print the observation dimensions
            
            this->mean_.setZero(this->obs_size_);
            this->var_.setZero(this->obs_size_);
            this->recent_mean_.setZero(this->obs_size_);
            this->recent_var_.setZero(this->obs_size_);
            this->delta_.setZero(this->obs_size_);
            this->epsilon_.setZero(this->obs_size_);
            this->epsilon_.setConstant(1e-8);
            // Pre allocate the memory for the observations vector
            this->dones_.setZero(this->num_envs_);
            this->rewards_.setZero(this->num_envs_);
            this->traversability_.setZero(this->num_envs_);
            this->froude_.setZero(this->num_envs_);
            this->projected_speed_.setZero(this->num_envs_);
            this->max_torque_.setZero(this->num_envs_);
            this->power_.setZero(this->num_envs_);
            this->observations_.setZero(this->num_envs_, this->obs_size_);     
               
        }

        /**
         * @brief Restart all environments
         *
         * @param epoch Current train epoch
         * @return std::vector<step_t>  Information returned in each environment
         */
        step_array_info_t reset()
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < this->num_envs_; i++)
            {   
                step_t step_info = this->environments_[i]->reset();
                this->observations_.row(i) = step_info.observation;
                this->dones_(i) = step_info.done;
                this->rewards_(i) = step_info.reward;
                this->traversability_(i) = step_info.info["traversability"];
                this->froude_(i) = step_info.info["froude"];
                this->projected_speed_(i) = step_info.info["projected_speed"];
                this->max_torque_(i) = step_info.info["max_torque"];
                this->power_(i) = step_info.info["power"];


            }
            // return the observations segmented by type
            return {
                // Non privileged observations
                this->observations_.block(0, this->regular_obs_begin_idx_, this->num_envs_, this->regular_obs_size_),
                // Privileged observations
                this->observations_.block(0, this->privileged_obs_begin_idx_, this->num_envs_, this->privileged_obs_size_),
                // Historic observations
                this->observations_.block(0, this->historic_obs_begin_idx_, this->num_envs_, this->historic_obs_size_),
                // Rewards
                this->rewards_,
                // Dones
                this->dones_,
                // Traversability
                this->traversability_,
                // Froude
                this->froude_,
                // Projected speed
                this->projected_speed_,
                // Max torque
                this->max_torque_,
                // Power
                this->power_
            };
            
        }

        /**
         * @brief Perform one step in each simulation
         *
         * @param actions Action taken in each environment
         * @return step_array_info_t Information returned in each environment
         */
        step_array_info_t step(Eigen::Ref<EigenRowMajorMat> &actions, bool update_scaling_stats)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < this->num_envs_; i++)
            {
                step_t step_info = this->environments_[i]->step(actions.row(i));
                this->observations_.row(i) = step_info.observation;
                this->dones_(i) = step_info.done;
                this->rewards_(i) = step_info.reward;
                this->traversability_(i) = step_info.info["traversability"];
                this->froude_(i) = step_info.info["froude"];
                this->projected_speed_(i) = step_info.info["projected_speed"];
                this->max_torque_(i) = step_info.info["max_torque"];
                this->power_(i) = step_info.info["power"];
                
                if (step_info.done && this->auto_reset)
                {
                    this->environments_[i]->reset();
                }
            }
            
            if (this->normalize_){
                this->update_statistics(this->observations_, update_scaling_stats);
            }
            return {
                // Non privileged observations
                this->observations_.block(0, this->regular_obs_begin_idx_, this->num_envs_, this->regular_obs_size_),
                // Privileged observations
                this->observations_.block(0, this->privileged_obs_begin_idx_, this->num_envs_, this->privileged_obs_size_),
                // Historic observations
                this->observations_.block(0, this->historic_obs_begin_idx_, this->num_envs_, this->historic_obs_size_),
                // Rewards
                this->rewards_,
                // Dones
                this->dones_,
                // Traversability
                this->traversability_,
                // Froude
                this->froude_,
                // Projected speed
                this->projected_speed_,
                // Max torque
                this->max_torque_,
                // Power
                this->power_
            };
        }

        /**
         * @brief Obtains the statistics of the observations
         *
         * @return statistics_t Observation statistics data
         */
        statistics_t get_statistics(void)
        {
            return {this->mean_, this->var_, this->count_};
        }

        const std::string &get_resource_dir() const
        {
            return this->resource_dir_;
        }

        const std::string &get_cfg_string() const
        {
            return this->cfg_string_;
        }

        /**
         * @brief Set the statistics of the observations
         *
         * @param stats_data Observation statistics data
         */
        void set_statistics(
            const Eigen::VectorXd &mean,
            const Eigen::VectorXd &var,
            const float count 
        )
        {
            this->mean_ = mean;
            this->var_ = var;
            this->count_ = count;
        }

        /**
         * Gets the observation sizes
        */
       std::vector<int> get_obs_sizes(void) {
        return {
            this->regular_obs_size_,
            this->privileged_obs_size_,
            this->historic_obs_size_
        };
       }

        /**
         * @brief Create the terrain that contains hills.
         *
         * @param frequency How often each hill appears.
         * @param amplitude Height of the hills.
         * @param roughness Terrain roughness.
         */
        void hills(double frequencies, double amplitudes, double roughness)
        {
// If windows use static schedule
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->hills(frequencies, amplitudes, roughness);
        }

        /**
         * @brief Create the terrain that contains stairs.
         *
         * @param width Width of each step.
         * @param height Height of each step.
         */
        void stairs(double widths, double heights)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->stairs(widths, heights);
        }

        /**
         * @brief Create the terrain that contains stepped terrain
         *
         * @param frequency Frequency of the cellular noise
         * @param amplitude Scale to multiply the cellular noise
         */
        void cellular_steps(double frequencies, double amplitudes)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->cellular_steps(frequencies, amplitudes);
        }

        /**
         * @brief Generates a terrain made of steps (little square boxes)
         *
         * @param width Width of each of the steps [m]
         * @param height Amplitude of the steps[m]
         */
        void steps(double widths, double heights)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->steps(widths, heights);
        }

        /**
         * @brief Generates a terrain made of a slope.
         *
         * @param slope The slope of the slope [m]
         * @param roughness Terrain roughness.
         */
        void slope(double slope, double roughness)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->slope(slope, roughness);
        }

        /**
         * @brief Sets the robot command direction. This method is used when
         * the robot command type is external.
         *
         * @param direction_angle Angle to which the robot must move
         * @param turning_direction Turning direction: 1 for clockwise, -1
         * for counter-clockwise and to not rotate.
         * @param stop The robot must not move.
         */
        void set_command(double direction_angle, int turning_direction, bool stop)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->set_command(direction_angle, int(turning_direction), stop);
        }

        /**
         * @brief Returns the number of parallel environments
         *
         */
        int get_num_envs(void)
        {
            return num_envs_;
        }

        /**
         * @brief Updates the curriculum coefficient
         *
         */
        void update_curriculum_coefficient(void)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++){
                environments_[i]->update_curriculum_coefficient();
            }
        }
        

        /**
         * @brief Changes the current curriculum coefficient
         *
         */
        void set_curriculum_coefficient(double new_coefficient)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_curriculum_coefficient(new_coefficient);
            }
        }

        std::map<std::string, std::array<int, 2>> get_observations_indexes(){
            return this->environments_[0]->get_observations_indexes();
        }
        


        /**
         * @brief Allows to place the robot in a specific position
         *
         * @param x Absolute x position
         * @param y Absolute y position
         * @param z Absolute z position
         * @param pitch Pitch angle position
         * @param yaw Yaw angle position
         * @param roll Roll angle position
         *
         */
        void set_absolute_position(
            double x,
            double y,
            double z,
            double roll,
            double pitch,
            double yaw
            )
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->set_absolute_position(x, y, z, roll, pitch, yaw);
        }

        /**
         * @brief Allows to set an absolute speed to the robot
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
            double angular_z)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->set_absolute_velocity(
                    linear_x,
                    linear_y,
                    linear_z,
                    angular_x,
                    angular_y,
                    angular_z);
        }

        /**
         * @brief Allows to place the robot in a specific position
         *
         * @param x Absolute x position
         * @param y Absolute y position
         * @param z Absolute z position
         * @param roll Roll angle position
         * @param pitch Pitch angle position
         * @param yaw Yaw angle position
         *
         */
        void set_foot_positions_and_base_pose(
            Eigen::Ref<EigenVec> &foot_positions,
            double x,
            double y,
            double z,
            double roll,
            double pitch,
            double yaw
            ){
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_foot_positions_and_base_pose(foot_positions, x, y, z, roll, pitch, yaw);
            }
        }

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
        ){
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_gait_config(
                     base_frequency,
                     leg_1_phase,
                     leg_2_phase,
                     leg_3_phase,
                     leg_4_phase,
                     foot_vertical_span,
                     angular_movement_delta,
                     x_movement_delta,
                     y_movement_delta,
                     leg_span,
                     use_horizontal_frame  
                    );
            }
        }
    };


    class NormalDistribution
    {
    public:
        NormalDistribution() : normDist_(0.f, 1.f) {}

        float sample() { return normDist_(gen_); }
        void seed(int i) { gen_.seed(i); }

    private:
        std::normal_distribution<float> normDist_;
        static thread_local std::mt19937 gen_;
    };
    thread_local std::mt19937 raisim::NormalDistribution::gen_;

    class NormalSampler
    {
    public:
        NormalSampler(int dim)
        {
            dim_ = dim;
            normal_.resize(THREAD_COUNT);
            seed(0);
        }

        void seed(int seed)
        {
// This ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < THREAD_COUNT; i++)
                normal_[0].seed(i + seed);
        }

        inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                           Eigen::Ref<EigenVec> &std,
                           Eigen::Ref<EigenRowMajorMat> &samples,
                           Eigen::Ref<EigenVec> &log_prob)
        {
            int agentNumber = int(log_prob.rows());

#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int agentId = 0; agentId < agentNumber; agentId++)
            {
                log_prob(agentId) = 0;
                for (int i = 0; i < dim_; i++)
                {
                    const float noise = normal_[omp_get_thread_num()].sample();
                    samples(agentId, i) = mean(agentId, i) + noise * std(i);
                    log_prob(agentId) -= float(noise * noise * 0.5 + std::log(std(i)));
                }
                log_prob(agentId) -= float(dim_) * 0.9189385332f;
            }
        }
        int dim_;
        std::vector<NormalDistribution> normal_;
    };

}
