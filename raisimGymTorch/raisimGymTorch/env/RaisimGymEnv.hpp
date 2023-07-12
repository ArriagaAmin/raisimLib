//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

namespace raisim
{

    /**
     * @brief Type that represents the output of a step in the simulation
     *
     */
    struct step_t
    {
        // Observing the state of the environment after applying the action.
        Eigen::VectorXd observation;
        // The reward obtained for applying the action.
        double reward;
        // A boolean value indicating whether or not the episode has ended.
        bool done;
        // Other relevant information about the state of the environment.
        std::map<std::string, double> info;
    };

    class RaisimGymEnv
    {
    protected:
        // Environment where the simulation occurs.
        std::unique_ptr<raisim::World> world_;
        // Simulation differential time.
        double simulation_dt_ = 0.0025;
        // Controlator differential time. It must be greater than the
        // simulation time differential.
        double control_dt_ = 0.01;
        // Directory where the resources needed to build the environment
        // are located
        std::string resource_dir_;
        // Environment configuration file.
        Yaml::Node cfg_;
        // Observation space dimensions.
        std::map<std::string, int> observation_dimensions_;
        // Action space dimension.
        int action_dimension_ = 0;
        // Pointer to the server running the simulation.
        std::unique_ptr<raisim::RaisimServer> server_;
        // Agent earned rewards.
        raisim::Reward rewards_;

    public:
        /**
         * @param resource_dir Directory where the resources needed to build
         * the environment are located
         * @param cfg Environment configuration file
         * @param visualizable Indicates if the robot target will be
         * displayed.
         * @param port Port through which the simulation will be displayed
         *
         */
        explicit RaisimGymEnv(
            std::string resource_dir,
            const Yaml::Node &cfg,
            bool visualizable,
            std::vector<std::string> non_privileged_obs,
            std::vector<std::string> privileged_obs,
            std::vector<std::string> historic_obs,
            int port = 8080) : resource_dir_(std::move(resource_dir)),
                               cfg_(cfg)
        {
        }

        virtual ~RaisimGymEnv()
        {
            if (server_)
                server_->killServer();
        };

        /**
         * @brief Resets the simulation to the initial state.
         *
         * @param epoch Current train epoch
         * @return step_t Current environment information
         */
        virtual step_t reset(int epoch) = 0;

        /**
         * @brief Perform a time step within the simulation.
         *
         * @param action Action taken by the robot.
         *
         * @return Environment information after applying the action
         *
         */
        virtual step_t step(const Eigen::Ref<EigenVec> &action) = 0;

        /**
         * @brief Sets the robot command direction. This method is used when
         * the robot command type is external.
         *
         * @param target_angle Angle to which the robot must move
         * @param turning_direction Turning direction: 1 for clockwise, -1
         * for counter-clockwise and to not rotate.
         * @param stop The robot must not move.
         */
        virtual void set_command(
            double target_angle, 
            int turning_direction, 
            bool stop) = 0;
    };
}
