//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "omp.h"
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
        std::map<std::string, Eigen::VectorXd> mean;
        // Variance of each observation
        std::map<std::string, Eigen::VectorXd> var;
        // Total samples performed
        float count;
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
        // Mean of each observation
        std::map<std::string, Eigen::VectorXd> mean_;
        // Variance of each observation
        std::map<std::string, Eigen::VectorXd> var_;
        // Total samples performed
        float count_ = 1e-4f;

        void update_statistics(Eigen::Ref<EigenMapVec> &observations, bool update)
        {
            // Variables used to calculate the mean and variance of
            // each observation
            std::map<std::string, Eigen::VectorXd> recent_mean, recent_var, delta;

            if (update)
            {
                // Calculate the mean and variance per position for each
                // key in each map
                std::map<std::string, Eigen::VectorXd> sum_squares;

                for (int i = 0; i < this->num_envs_; i++)
                {
                    for (const auto &pair : observations(i, 0))
                    {
                        const std::string &key = pair.first;
                        const Eigen::VectorXd &vec = pair.second;
                        if (recent_mean.count(key) == 0)
                        {
                            recent_mean[key] = vec;
                            sum_squares[key] = vec.array().square().matrix();
                        }
                        else
                        {
                            recent_mean[key] += vec;
                            sum_squares[key] += vec.array().square().matrix();
                        }
                    }
                }
                for (auto &pair : recent_mean)
                {
                    pair.second /= observations.rows();
                }
                for (auto &pair : sum_squares)
                {
                    pair.second /= observations.rows();
                }

                recent_var.clear();
                for (const auto &pair : recent_mean)
                {
                    const std::string &key = pair.first;
                    const Eigen::VectorXd &mean = pair.second;
                    recent_var[key] = sum_squares[key] - mean.array().square().matrix();
                }

                // Calculate the delta
                for (const auto &pair : this->mean_)
                {
                    const std::string &key = pair.first;
                    const Eigen::VectorXd &mean = pair.second;
                    const Eigen::VectorXd &current_recent_mean = recent_mean[key];
                    delta[key] = (mean - current_recent_mean).array().square().matrix();
                }

                // Update the cumulative mean and variance
                float total_count = this->count_ + this->num_envs_;
                for (auto &pair : this->mean_)
                {
                    const std::string &key = pair.first;
                    Eigen::VectorXd &mean = pair.second;
                    const Eigen::VectorXd &current_recent_mean = recent_mean[key];
                    mean = mean * (this->count_ / total_count) +
                           current_recent_mean * (this->num_envs_ / total_count);
                }
                for (auto &pair : this->var_)
                {
                    const std::string &key = pair.first;
                    Eigen::VectorXd &var = pair.second;
                    const Eigen::VectorXd &current_recent_var = recent_var[key];
                    const Eigen::VectorXd &current_delta = delta[key];
                    var = (var * this->count_ +
                           current_recent_var * this->num_envs_ +
                           current_delta * (this->count_ * this->num_envs_ / total_count)) /
                          total_count;
                }
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
                std::map<std::string, Eigen::VectorXd> &map = observations(i, 0);
                for (auto &pair : map)
                {
                    const std::string &key = pair.first;
                    Eigen::VectorXd &vec = pair.second;
                    const Eigen::VectorXd &mean = mean_[key];
                    const Eigen::VectorXd &var = var_[key];
                    Eigen::VectorXd epsilon = Eigen::VectorXd::Constant(vec.size(), 1e-8);
                    vec = (vec.array() - mean.array()) / (var.array() + epsilon.array()).sqrt();
                }
            }
        }

    public:
        explicit VectorizedEnvironment(
            std::string resource_dir,
            std::string cfg,
            int port,
            bool normalize = true) : resource_dir_(resource_dir),
                                     cfg_string_(cfg),
                                     normalize_(normalize),
                                     port_(port)
        {
            Yaml::Parse(this->cfg_, cfg);

            this->auto_reset = this->cfg_["control"]["auto_reset"].template As<bool>();
            if (&this->cfg_["simulation"]["render"])
                this->render_ = this->cfg_["simulation"]["render"].template As<bool>();
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
            RSINFO("ENVIRONMENTS COUNT: \033[1m" + std::to_string(this->num_envs_) + "\033[0m.");

            this->environments_.reserve(this->num_envs_);
            RSINFO("Creating environments.");
            for (int i = 0; i < this->num_envs_; i++)
            {
                this->environments_.push_back(new ChildEnvironment(
                    this->resource_dir_,
                    this->cfg_,
                    this->render_ && i == 0,
                    this->port_));
            }

            this->observation_dimensions_ = this->environments_[0]->get_observations_dimension();
            this->action_dim = this->environments_[0]->get_action_dimension();

            RSINFO("Normalize: " + std::to_string(this->normalize_));
            if (this->normalize_)
            {
                for (const auto &[key, value] : this->observation_dimensions_)
                {
                    this->mean_[key] = Eigen::VectorXd::Zero(value);
                    this->var_[key] = Eigen::VectorXd::Zero(value);
                }
            }
        }

        /**
         * @brief Restart all environments
         *
         * @param epoch Current train epoch
         * @return std::vector<step_t>  Information returned in each environment
         */
        std::vector<step_t> reset(int epoch)
        {
            std::vector<step_t> result(this->num_envs_);
            this->epoch_ = epoch;

#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < this->num_envs_; i++)
            {
                step_t step_info = this->environments_[i]->reset(this->epoch_);
                result[i] = step_info;
            }
            return result;
        }

        /**
         * @brief Perform one step in each simulation
         *
         * @param actions Action taken in each environment
         * @return std::vector<step_t>  Information returned in each environment
         */
        std::vector<step_t> step(Eigen::Ref<EigenRowMajorMat> &actions)
        {
            std::vector<step_t> result(this->num_envs_);

#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < this->num_envs_; i++)
            {
                step_t step_info = this->environments_[i]->step(actions.row(i));
                result[i] = step_info;
                if (step_info.done && this->auto_reset)
                {
                    this->environments_[i]->reset(this->epoch_);
                }
            }
            return result;
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
        void set_statistics(statistics_t stats_data)
        {
            this->mean_ = stats_data.mean;
            this->var_ = stats_data.var;
            this->count_ = stats_data.count;
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
        void set_command(double direction_angle, double turning_direction, bool stop)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->set_command(direction_angle, turning_direction, stop);
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
            double pitch,
            double yaw,
            double roll)
        {
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->set_absolute_position(x, y, z, pitch, yaw, roll);
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
            int agentNumber = log_prob.rows();

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
                    log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
                }
                log_prob(agentId) -= float(dim_) * 0.9189385332f;
            }
        }
        int dim_;
        std::vector<NormalDistribution> normal_;
    };

}
