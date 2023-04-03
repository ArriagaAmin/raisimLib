#include "ExternalForce.hpp"

ExternalForceApplier::ExternalForceApplier(
    raisim::ArticulatedSystem *anymal,
    double time_step_threshold,
    double external_force_maximun_magnitude) : anymal_(anymal),
                                               time_threshold_(time_step_threshold),
                                               maximun_magnitude_(external_force_maximun_magnitude)
{
    // Using a uniform random distribution to generate a random force
    // from 0 to external_force_maximun_magnitude
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    Eigen::Vector3d external_force_direction;

    double az, el;
    az = dis(gen) * M_PI;
    el = dis(gen) * M_PI / 2;
    external_force_direction << cos(az) * cos(el), sin(az) * cos(el), sin(el);

    this->external_force_world_frame = external_force_direction *
                                       dis(gen) * external_force_maximun_magnitude;
};

void ExternalForceApplier::apply_external_force(double time_step)
{
    if (time_step <= this->time_threshold_)
    {
        this->anymal_->setExternalForce(0, this->external_force_world_frame);
    }
    else
    {
        this->external_force_world_frame = Eigen::Vector3d::Zero();
    }
};

void ExternalForceApplier::external_force_in_base(Eigen::Matrix3d R_base_to_world)
{
    this->external_force_base_frame = R_base_to_world * this->external_force_world_frame;
}
