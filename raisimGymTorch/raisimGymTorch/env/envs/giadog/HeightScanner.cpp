#include "HeightScanner.hpp"

HeightScanner::HeightScanner(
    raisim::World *world,
    raisim::ArticulatedSystem *anymal,
    const EnvConfig *env_config,
    std::vector<std::string> feet_link_names,
    bool visualizable) : world_(world),
                                 anymal_(anymal),
                                 feet_link_names_(feet_link_names),
                                 visualizable_(visualizable)
{
    this->n_scan_rings_ = env_config->N_SCAN_RINGS;
    this->scans_per_ring_ = env_config->SCANS_PER_RING;
    this->foot_scan_radius_ = env_config->FOOT_SCAN_RADIUS;
    this->n_legs_ = static_cast<int>(feet_link_names_.size());
    this->feet_height_scan.setZero(n_scans_);
    this->current_feet_position.setZero(3 * n_legs_);
    this->scans_per_foot_ = scans_per_ring_ * n_scan_rings_ + 1;
    this->n_scans_ = this->scans_per_foot_ * n_legs_;

    for (int i = 0; i < feet_link_names.size(); i++)
    {
        this->feet_frames_idx_.push_back(static_cast<int>(
            this->anymal_->getFrameIdxByName(feet_link_names_[i])
            ));
    };
};

std::vector<Eigen::Vector2d> HeightScanner::footScanCoordinates_(
    double x,
    double y,
    double yaw)
{
    // Foot scan radius
    double r = this->foot_scan_radius_;
    double yaw_i;
    double phi = 2 * M_PI / this->scans_per_ring_;
    Eigen::Vector2d scan_coordinates;
    std::vector<Eigen::Vector2d> foot_scan_coordinates;

    // The first scan is directly below the foot
    scan_coordinates[0] = x;
    scan_coordinates[1] = y;
    foot_scan_coordinates.push_back(scan_coordinates);

    // The other scans are spaced evenly around the foot
    for (int j = 0; j < this->n_scan_rings_; j++)
    {
        for (int i = 0; i < this->scans_per_ring_; i++)
        {
            yaw_i = yaw + i * phi;
            scan_coordinates[0] = x + r * std::cos(yaw_i);
            scan_coordinates[1] = y + r * std::sin(yaw_i);

            foot_scan_coordinates.push_back(scan_coordinates);
        };

        r = r + this->foot_scan_radius_;
    };

    return foot_scan_coordinates;
};

void HeightScanner::foot_scan(double base_yaw)
{
    std::vector<Eigen::Vector2d> points;

    for (int i = 0; i < this->n_legs_; i++)
    {
        int foot_frame_index = this->feet_frames_idx_[i];

        // Create a raisim vector to store the foot position
        raisim::Vec<3> foot_position;
        // Get the foot position in world frame
        this->anymal_->getFramePosition(foot_frame_index, foot_position);
        // Cast the vector to a Eigen vector
        Eigen::Vector3d foot_position_eigen = foot_position.e();
        this->current_feet_position.segment<3>(3 * i) = foot_position_eigen;

        // We get the square of the speed of the legs.
        raisim::Vec<3> foot_velocity;
        this->anymal_->getFrameVelocity(foot_frame_index, foot_velocity);
        this->feet_speed_squared_[i] = foot_velocity.e().squaredNorm();

        // Calculate the foot scan coordinates
        points = this->footScanCoordinates_(
            foot_position_eigen[0],
            foot_position_eigen[1],
            base_yaw);

        // Perform the height scan for each point
        for (int j = 0; j < (this->scans_per_foot_); j++)
        {
            // We perform a raycast to get the height of the ground around the
            // x, y cordinates of each point. The raytest is performed in order
            // to colide only with the ground and not with the robot
            const raisim::RayCollisionList &col = this->world_->rayTest(
                {points[j][0], points[j][1], 10.0},   // Start point
                {0.0, 0.0, -1.0},                     // Direction
                50.,                                  // Max distance
                true,                                 // Return all hits
                0,                                    // Ignore this body id
                0,                                    // Ignore this body id
                raisim::RAISIM_STATIC_COLLISION_GROUP // Collision group
            );

            // Calculate the heaigh scan value as the difference between the
            // ground height and the scan point height
            this->feet_height_scan[i * (this->scans_per_foot_) + j] = (col[0].getPosition()[2] - foot_position_eigen[2]);

            // Update the visual shapes if the height scan is visualizable
            if (this->visualizable_)
            {
                int sphere_index = i * this->scans_per_foot_ + j;
                raisim::Visuals *sphere = this->foot_height_scan_visuals_[sphere_index];
                sphere->setPosition(
                    points[j][0],
                    points[j][1],
                    col[0].getPosition()[2]);
            };
        };
    };
};

double HeightScanner::clearance_reward(const Eigen::Vector4d &foot_phases)
{
    this->foot_clearance_reward = 0;
    int n_swing_feet = 0;

    for (int i = 0; i < this->n_legs_; i++)
    {
        double phi = foot_phases[i];
        if (M_PI * 2 > phi && phi > M_PI)
        {
            n_swing_feet++;
            double max_height = this->feet_height_scan.segment(
                                                          i * this->scans_per_foot_,
                                                          this->scans_per_foot_)
                                    .maxCoeff();
            this->foot_clearance_reward += (max_height > 0.07) ? 1.0 : 0.0;
        }
    }
    if (n_swing_feet > 0)
    {
        this->foot_clearance_reward /= n_swing_feet;
    }

    return this->foot_clearance_reward;
}

void HeightScanner::add_visual_indicators(raisim::RaisimServer *server)
{
    // Add to the height_scan_visuals_ vector eight speheres to represent the
    // height of the ground.
    if (this->visualizable_)
    {
        for (int i = 0; i < n_scans_; i++)
        {
            this->foot_height_scan_visuals_.push_back(
                server->addVisualSphere(
                    "height_scan_" + std::to_string(i),
                    0.01,  // radius
                    0.294, // red
                    0,     // green
                    0.5,   // blue
                    1      // alpha
                    ));
        }
    }
}