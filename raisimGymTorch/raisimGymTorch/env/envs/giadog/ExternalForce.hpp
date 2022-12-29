#include "RaisimGymEnv.hpp"

/**
 * @brief Class that is used to apply an external force to the robot base, 
 * and to get the external force expresed in the robot base frame.
*/
class ExternalForceApplier
{
    private:
        // Robot that is in the simulation.
        raisim::ArticulatedSystem* anymal_;
        // Time during which the external force will be applied to the robot.
        double time_threshold_;
        // Maximum magnitude of the external force applied to the robot.
        double maximun_magnitude_;

    public: 
        // External force applied to the robot.
        Eigen::Vector3d external_force_world_frame, external_force_base_frame;
   
        explicit ExternalForceApplier(void) { };

        /**
         * @param anymal Robot that is in the simulation.
         * @param time_step_threshold Time during which the external force will
         *      be applied to the robot.
         * @param external_force_maximun_magnitude Maximum posible magnitude 
         *      of the external force.
         */
        explicit ExternalForceApplier(
            raisim::ArticulatedSystem* anymal,
            double time_step_threshold,
            double external_force_maximun_magnitude
        ) : anymal_(anymal),
            time_threshold_(time_step_threshold),
            maximun_magnitude_(external_force_maximun_magnitude)
        { 
            // Using a uniform random distribution to generate a random force
            // from 0 to external_force_maximun_magnitude
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double>  dis(-1.0, 1.0);
            
            Eigen::Vector3d external_force_direction;

            double az, el;
            az = dis(gen) * M_PI;
            el = dis(gen) * M_PI/2;
            external_force_direction << cos(az)*cos(el), sin(az)*cos(el), sin(el);
            
            this->external_force_world_frame = external_force_direction * 
                dis(gen) * external_force_maximun_magnitude;
        };  

        /**
         * @brief Applies the external force to the robot base
         * 
         * @param time_step Current time step in the environment.
         * 
         */
        void applyExternalForce(double time_step);


        /**
         * @brief Gets the external force applied to the robot base in the base 
         * frame
         * 
         * @param R_base_to_world The rotation matrix from the base frame to 
         *      the world frame.
         *
         */
        void externalForceBaseFrame(Eigen::Matrix3d R_base_to_world);

};


void ExternalForceApplier::applyExternalForce(double time_step){
    if (time_step <= this->time_threshold_)
    {
        this->anymal_->setExternalForce(0, this->external_force_world_frame);
    }
    else 
    {
        this->external_force_world_frame = Eigen::Vector3d::Zero();
    }    
};

void ExternalForceApplier::externalForceBaseFrame(Eigen::Matrix3d R_base_to_world)
{
    this->external_force_base_frame = R_base_to_world * this->external_force_world_frame;
}


