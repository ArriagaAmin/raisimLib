//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

namespace raisim {


class RaisimGymEnv {
     protected:
        // Environment where the simulation occurs.
        std::unique_ptr<raisim::World> world_;
        // Simulation differential time.
        double simulation_dt_ = 0.001;
        // Controlator differential time. It must be greater than the 
        // simulation time differential.
        double control_dt_ = 0.01;
        // Directory where the resources needed to build the environment 
        // are located.
        std::string resource_dir_;
        // Environment configuration file.
        Yaml::Node cfg_;
        // Observation space dimension.
        int ob_dim_ = 0;
        // Action space dimension.
        int action_dim_ = 0;
        // Pointer to the server running the simulation.
        std::unique_ptr<raisim::RaisimServer> server_;
        // Agent earned rewards.
        raisim::Reward rewards_;

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
        explicit RaisimGymEnv (
            std::string resource_dir, 
            const Yaml::Node& cfg,
            bool visualizable,
            int port=8080
        ) :
            resource_dir_(std::move(resource_dir)), 
            cfg_(cfg)
        { }

        virtual ~RaisimGymEnv() { if(server_) server_->killServer(); };

        ////////////////////////// Mandatory methods //////////////////////////
        /**
         * @brief initialize the environment
         */
        virtual void init(void) = 0;

        /**
         * @brief Resets the simulation to the initial state.
         * 
         */
        virtual void reset(void) = 0;

        /**
         * @brief Updates the robot's observations vector.
         * 
         * @param ob Vector to contain the observations.
         * 
         */
        virtual void observe(Eigen::Ref<EigenVec> ob) = 0;

        /**
         * @brief Perform a time step within the simulation.
         * 
         * @param action Action taken by the robot.
         * 
         * @return Reward obtained in this time step.
         * 
         */
        virtual float step(const Eigen::Ref<EigenVec>& action) = 0;

        /**
         * @brief Check if the current state of the robot is terminal, 
         * that is, if the robot fell or reached its goal
         * 
         * @param terminalReward 
         * 
         * @return Indicates if the current state of the robot is terminal
         */
        virtual bool isTerminalState(void) = 0;
        //////////////////////////////////////////////////////////////////////


        ////////////////////////// Optional methods //////////////////////////
        /**
         * @brief Get the Traversability object
         * 
         * @param trav 
         */
        virtual double getTraversability(void) = 0;

        /**
         * @brief Returns the power used by the robot.
         * 
         * 
         */
        virtual double getPower(void) = 0;

        /**
         * @brief Returns the froude number of the robot.
         * 
         */
        virtual double getFroude(void) = 0;

        /**
         * @brief Returns the robot orthogonal speed [m/s]
         * 
         */
        virtual double getProjSpeed(void) = 0;

        /**
         * @brief Returns the maximun torque applied by the motors of the robot. [N,m]
         * 
         */
        virtual double getMaxTorque(void) = 0;

        /**
         * @brief Sets the robot command direction.
         * 
         */
        virtual void setCommand(
            double direction_angle, 
            double turning_direction, 
            bool stop
        ) = 0;

        /**
         * @brief Sets the robot PD gains. Depending on the configuration 
         * file it can be a default value or a random value from a uniform 
         * distribution. Check the configuration file for more information 
         * (cfg.yaml).
         * 
         */
        virtual void setPDGains(void) = 0;
        //////////////////////////////////////////////////////////////////////
};
}

#endif //SRC_RAISIMGYMENV_HPP
