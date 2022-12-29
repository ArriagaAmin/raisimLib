#include <Eigen/Dense>
#include "EnvConfig.hpp"

/**
 * @brief Position delta for placement heuristics.
 * 
 * @param ftg_freqs Foot Trajectory Generator frequencies.
 * @param ftg_sine_phases Foot Trajectory Generator phases.
 * @param command_dir Robot command direction.
 * @param turn_dir Robot rotation direction.
 * @param add_cartesian_delta Cartesian foot placement heuristic enabled/disabled.
 * @param add_angular_delta Angular foot placement heuristic enabled/disabled.
 * @param config Simulation environment configuration parameters.
 * 
 * @return Position delta for placement heuristics.
 */
Eigen::MatrixXd directionDeltas(
    Eigen::Vector4d ftg_freqs, 
    Eigen::Vector4d ftg_sine_phases, 
    double command_dir, 
    int turn_dir,
    bool add_cartesian_delta,
    bool add_angular_delta,
    EnvConfig *config
) { 
    double LEG_SPAN =config->LEG_SPAN; // Leg span
    double dt = config->CONTROL_DT; // Control time step
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(4, 3);

    for (int i = 0; i < 4; i++) 
    {
        Eigen::Vector3d position_delta(0.0, 0.0, 0.0);
        
        // Position delta
        position_delta(0) = 1.7 * cos(command_dir) * dt * ftg_sine_phases(i) * 
            ftg_freqs(i) * LEG_SPAN;
        position_delta(1) = 1.02 * sin(command_dir) * dt * ftg_sine_phases(i) *
            ftg_freqs(i) * LEG_SPAN;

        // Rotation delta 
        Eigen::Vector3d rotation_delta(0.0, 0.0, 0.0);
        
        double theta = M_PI/4;
        double phi_arc = (i == 0) * -theta + (i == 1) * -(M_PI - theta) + 
            (i == 2) *  theta + (i == 3) * (M_PI - theta);
        
        rotation_delta(0) = 0.68 * -cos(phi_arc) * dt * ftg_sine_phases(i) * 
            turn_dir * ftg_freqs(i) * LEG_SPAN;
        rotation_delta(1) = 0.68 * -sin(phi_arc) * dt * ftg_sine_phases(i) * 
            turn_dir * ftg_freqs(i) * LEG_SPAN;
        
        delta.row(i) += (
            position_delta * add_cartesian_delta + 
            rotation_delta * add_cartesian_delta
        );
    }
    return delta;
}