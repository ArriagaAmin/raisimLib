#include <vector>
#include <string>
#include <Eigen/Dense>
#include "RaisimGymEnv.hpp"

/**
 * @brief Class thats is used to get the contcat states of the robot leg 
 * articulations, the friction coeficients of the legs and the forces and
 * normal vectors to the robot feet. The contact states are stored in the 
 * attributes of the class. It also calculates the undesirable_collisions_reward_
 * which is used to penalize the collision of the robot shanks and thighs 
 * with the ground.
 */ 
class ContactSolver
{
    private:
        // Differential time.
        double dt_; 
        // Environment where the simulation occurs.
        raisim::World* world_;
        // Robot that is in the simulation.
        raisim::ArticulatedSystem* anymal_;
        // Id of objects that is in the simulation.
        std::vector<int> shank_ids_, thigh_ids_, foot_ids_, foot_links_ids_;

    public:
        // Direction of the normal force on each leg of the robot.
        Eigen::VectorXd terrain_normal; 
        // Status of contacts on parts of the robot.
        Eigen::Vector4d thigh_contact_states, 
            shank_contact_states, 
            foot_contact_states;
        // Contact force of robot legs.
        Eigen::Vector4d foot_contact_forces;
        // Friction force on the robot's legs.
        Eigen::Vector4d foot_ground_friction;
        // Reward obtained for undesirable collisions.
        double undesirable_collisions_reward_;


        explicit ContactSolver(void) { };

        /**
         * @brief Construct a new Contact And Fricction Info Solver object.
         * 
         * @param world Pointer to the Environment.
         * @param anymal Pointer to the Robot that is in the Environment.
         * @param simulation_dt The simulation time step (Timestep of the 
         *      raisim simulation, not the control loop time step).
         * @param fricction_coeff_mean Mean of the fricction coefficient
         *      between the foot and the terrain.
         * @param fricction_coeff_std  Standard deviation of the fricction 
         *      coefficient between the foot and the terrain.
         * @param thigh_names List of the names of the thigh articulations 
         *      objects of the robot.
         * @param shank_names List of the names of the shank articulations 
         *      objects of the robot.
         * @param foot_names  List of the names of the foot articulations 
         *      objects of the robot.
         * 
         * @details The thigh_names, shank_names and foot_names are the names 
         * of the frames of the robot. There is a distinction in raisim between 
         * the links, the frames and the joints. The order of the links is:
         *  * Front left, 
         *  * Front right, 
         *  * Back left, 
         *  * Back right
         */
        explicit ContactSolver(
            raisim::World* world, 
            raisim::ArticulatedSystem* anymal,
            double simulation_dt,
            double fricction_coeff_mean,
            double fricction_coeff_std,
            std::vector<std::string> thigh_names,
            std::vector<std::string> shank_names,
            std::vector<std::string> foot_names
        ) : world_(world), anymal_(anymal), dt_(simulation_dt)
        { 
            this->dt_ = simulation_dt;

            // Cycle through the names and get the ids
            for(int i = 0; i < thigh_names.size(); i++)
            {
                thigh_ids_.push_back(this->anymal_->getBodyIdx(thigh_names[i])); 
            };
            for(int i = 0; i < shank_names.size(); i++)
            {
                shank_ids_.push_back(this->anymal_->getBodyIdx(shank_names[i]));
            };
            for(int i = 0; i < foot_names.size(); i++)
            {
                foot_ids_.push_back(this->anymal_->getBodyIdx(foot_names[i]));
            };

            // Use a random distribution to generate the friction coefficients
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dis(
                fricction_coeff_mean, 
                fricction_coeff_std
            );
            double foot_fricction_coeff; 
            this->foot_ground_friction.setZero(foot_ids_.size());
            
            // This is a little bit of a hack, but it works.
            // If you are going to use another robot, you will need to change 
            // this. Use the this->anymal_->getCollisionBodies() to get the 
            // collision bodies. Then use the col_body.colObj->name to get the 
            // name of the body. Ussually the names are self explanatory, but 
            // if you are using a different robot, you should check the names 
            // in the urdf file.
            int leg_index = 0;  
            for (raisim::CollisionDefinition col_body: this->anymal_->getCollisionBodies())
            {   
                std::string col_body_name;
                col_body_name = col_body.colObj->name;
                int pos = col_body_name.find("/");
                std::string col_type = col_body_name.substr(pos - 4 , 4);

                if (col_type == "foot")
                {
                    col_body.setMaterial(col_body_name);

                    foot_fricction_coeff = dis(gen);
                    this->world_->setMaterialPairProp(
                        "ground", 
                        col_body_name, 
                        foot_fricction_coeff , 
                        0, 
                        0
                    );
                    this->foot_ground_friction[leg_index] = foot_fricction_coeff;
                    leg_index++;
                }
                else
                {
                    col_body.setMaterial("plastic");
                }    
            }

            // We set the friction coefficient between the ground and the other 
            // robot collision bodies to be the same as the raisim default.
            // (0.8, 0.0, 0.0)
            this->world_->setMaterialPairProp("ground", "plastic", 0.8, 0.0, 0.0);

            // Set the vectors to zero
            this->thigh_contact_states.setZero(thigh_ids_.size());
            this->shank_contact_states.setZero(shank_ids_.size());
            this->foot_contact_states.setZero(foot_ids_.size());
            this->foot_contact_forces.setZero(foot_ids_.size());
            this->terrain_normal.setZero(foot_ids_.size() * 3);
        };

        /**
         * @brief Computes the contact states and contact forces for the legs.
         * This function is called every control time step.
         */
        void computeContactInfo(void);
};


void ContactSolver::computeContactInfo(void){
    this->thigh_contact_states.setZero(this->thigh_ids_.size());
    this->shank_contact_states.setZero(this->shank_ids_.size());
    this->foot_contact_states.setZero(this->foot_ids_.size());
    this->foot_contact_forces.setZero(this->foot_ids_.size());
    this->terrain_normal.setZero(this->foot_ids_.size() * 3);

    for(auto& contact: this->anymal_->getContacts()) {
        for(int i = 0; i < foot_ids_.size(); i++)
        {   
            // if there is no contact, skip the contact
            // (The vector is already zero, so we don't need to do anything)) 
            if (contact.skip()) continue; 
            
            // If not get the contact body id
            int contact_body_idx = contact.getlocalBodyIndex();
            
            // If it is one of the thigh or shank, then set the contact state to 1
            if(contact_body_idx == thigh_ids_[i])
            {      
                this->thigh_contact_states[i] = 1;   
            }
            else if(contact_body_idx == shank_ids_[i])
            {   
                this->shank_contact_states[i] = 1;  
            }
            // If it is one of the foot:
            // Set the contact state to 1
            // Get the contact force
            // Get the normal vector of the contact
            else if(contact_body_idx == foot_ids_[i])
            {   
                double norm_contact = contact.getImpulse().e().norm();
                
                this->foot_contact_states[i] = 1;
                this->foot_contact_forces[i] = norm_contact / this->dt_;  
                this->terrain_normal.segment<3>(i * 3) = contact.getNormal().e().transpose();
            }; 
        };
    };

    this->undesirable_collisions_reward_ = -(
        this->shank_contact_states.sum() + this->thigh_contact_states.sum()
    );  
};