#include "ContactSolver.hpp"

ContactSolver::ContactSolver(
    raisim::World *world,
    raisim::ArticulatedSystem *quadruped,
    double simulation_dt,
    double fricction_coeff_mean,
    double fricction_coeff_std,
    std::vector<std::string> thigh_parent_names,
    std::vector<std::string> shank_parent_names,
    std::vector<std::string> foot_parent_names,
    std::vector<std::string> foot_link_names
    ) : 
    world_(world), 
    quadruped_(quadruped), 
    dt_(simulation_dt)
{
    this->dt_ = simulation_dt;

    // Cycle through the names and get the ids
    // Note: We use the parent names because of the way raisim getBodyIdx method works.
    
    RSINFO("Setting the contact ids")
    for (int i = 0; i < thigh_parent_names.size(); i++)
    {
        thigh_ids_.push_back(static_cast<int>(
            this->quadruped_->getBodyIdx(thigh_parent_names[i])
            ));
    };
    for (int i = 0; i < shank_parent_names.size(); i++)
    {
        shank_ids_.push_back(static_cast<int>(
            this->quadruped_->getBodyIdx(shank_parent_names[i])
            ));
    };
    for (int i = 0; i < foot_parent_names.size(); i++)
    {
        foot_ids_.push_back(static_cast<int>(
            this->quadruped_->getBodyIdx(foot_parent_names[i]))
        );
    };

    // print the ids
    printf("Thigh ids: ");
    for (int i = 0; i < thigh_ids_.size(); i++)
    {
        printf("%d ", thigh_ids_[i]);
    }
    printf("\n");

    printf("Shank ids: ");
    for (int i = 0; i < shank_ids_.size(); i++)
    {
        printf("%d ", shank_ids_[i]);
    }
    printf("\n");

    printf("Foot ids: ");
    for (int i = 0; i < foot_ids_.size(); i++)
    {
        printf("%d ", foot_ids_[i]);
    }
    printf("\n");


    // Use a random distribution to generate the friction coefficients
    
    this->foot_ground_friction.setZero(static_cast<int>(foot_ids_.size()));

    this->foot_link_names = foot_link_names;

    // This is a little bit of a hack, but it works.
    // If you are going to use another robot, you will need to change
    // this. Use the this->quadruped_->getCollisionBodies() to get the
    // collision bodies. Then use the col_body.colObj->name to get the
    // name of the body. Ussually the names are self explanatory, but
    // if you are using a different robot, you should check the names
    // in the urdf file.
    // int leg_index = 0;
    // for (raisim::CollisionDefinition col_body : this->quadruped_->getCollisionBodies())
    // {   
    //     // TODO: This is a hack. Find a better way to do this.
    //     std::string col_body_name;
    //     col_body_name = col_body.colObj->name;
    //     int pos = static_cast<int>(col_body_name.find("/"));
    //     std::string col_type = col_body_name.substr(pos - 4, 4);

    //     if (col_type == "foot")
    //     {
    //         col_body.setMaterial(col_body_name);

    //         foot_fricction_coeff = dis(gen);
    //         this->world_->setMaterialPairProp(
    //             "ground",
    //             col_body_name,
    //             foot_fricction_coeff,
    //             0,
    //             0);
    //         this->foot_ground_friction[leg_index] = foot_fricction_coeff;
    //         leg_index++;
    //     }
    //     else
    //     {
    //         col_body.setMaterial("plastic");
    //     }
    // }

    
    RSINFO("Setting the foot friction")
    this->set_feet_friction(fricction_coeff_mean, fricction_coeff_std);


    // We set the friction coefficient between the ground and the other
    // robot collision bodies to be the same as the raisim default.
    // (0.8, 0.0, 0.0)
    //this->world_->setMaterialPairProp("ground", "plastic", 0.8, 0.0, 0.0);

    // Set the vectors to zero
    this->thigh_contact_states.setZero(thigh_ids_.size());
    this->shank_contact_states.setZero(shank_ids_.size());
    this->foot_contact_states.setZero(foot_ids_.size());
    this->foot_contact_forces.setZero(foot_ids_.size());
    this->terrain_normal.setZero(foot_ids_.size() * 3);
};

void ContactSolver::contact_info(void)
{
    this->thigh_contact_states.setZero(this->thigh_ids_.size());
    this->shank_contact_states.setZero(this->shank_ids_.size());
    this->foot_contact_states.setZero(this->foot_ids_.size());
    this->foot_contact_forces.setZero(this->foot_ids_.size());
    this->terrain_normal.setZero(this->foot_ids_.size() * 3);

    for (auto &contact : this->quadruped_->getContacts())
    {
        for (int i = 0; i < static_cast<int>(foot_ids_.size()); i++)
        {
            // if there is no contact, skip the contact
            // (The vector is already zero, so we don't need to do anything))
            if (contact.skip())
                continue;

            // If not get the contact body id
            int contact_body_idx = static_cast<int>(contact.getlocalBodyIndex());

            // If it is one of the thigh or shank, then set the contact state to 1
            if (contact_body_idx == thigh_ids_[i])
            {
                this->thigh_contact_states[i] = 1;
            }
            else if (contact_body_idx == shank_ids_[i])
            {
                this->shank_contact_states[i] = 1;
            }
            // If it is one of the foot:
            // Set the contact state to 1
            // Get the contact force
            // Get the normal vector of the contact
            else if (contact_body_idx == foot_ids_[i])
            {
                double norm_contact = contact.getImpulse().e().norm();

                this->foot_contact_states[i] = 1;
                this->foot_contact_forces[i] = norm_contact / this->dt_;
                this->terrain_normal.segment<3>(i * 3) = contact.getNormal().e().transpose();

            };
        };
    };

    this->undesirable_collisions_reward_ = -(
        this->shank_contact_states.sum() + this->thigh_contact_states.sum());
};

void ContactSolver::set_feet_friction(double fricction_coeff_mean, 
                                      double fricction_coeff_std)
                                    {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(
        fricction_coeff_mean,
        fricction_coeff_std);
    double foot_fricction_coeff;
    
    int leg_index = 0;
    for (std::string foot_name : this->foot_link_names)
    {   
        RSINFO("Setting friction for foot: " + foot_name)
        raisim::CollisionDefinition foot_col_body = this->quadruped_->getCollisionBody(foot_name + "/0");
        foot_fricction_coeff = dis(gen);
        foot_col_body.setMaterial(foot_name);
        this->world_->setMaterialPairProp(
                "ground",
                foot_name,
                foot_fricction_coeff,
                0,
                0);
        this->foot_ground_friction[leg_index] = foot_fricction_coeff;
        leg_index++;

    }
}