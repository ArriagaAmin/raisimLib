#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "EnvConfig.hpp"


/**
 * @brief Calculates the leg's Inverse kinematicks parameters:
 * The leg Domain 'D' (caps it in case of a breach) and the leg's radius.
 * 
 * @param x hip-to-foot distance in x-axis
 * @param y hip-to-foot distance in y-axis
 * @param z hip-to-foot distance in z-axis
 * @param config Simulation environment configuration parameters.
 * 
 * @return leg's Domain D.
 * @return leg's outer radius.
 */
std::pair<double, double> IKParams(
    double x, 
    double y, 
    double z, 
    EnvConfig *config
) {   
    double r_o, D, sqrt_component;

    sqrt_component = std::max(
        (double) 0.0, 
        std::pow(z, 2) + std::pow(y, 2) - std::pow(config->H_OFF, 2)
    );
    r_o = std::sqrt(sqrt_component) - config->V_OFF;
    D = (
        std::pow(r_o, 2) + std::pow(x, 2) - std::pow(config->SHANK_LEN, 2) - 
        std::pow(config->THIGH_LEN, 2)
    ) / (
        2 * config->SHANK_LEN * config->THIGH_LEN
    );
    D = std::max(std::min(D, 1.0),-1.0);

    return {D, r_o};
}

/**
 * @brief Right Leg Inverse Kinematics Solver
 * 
 * @param x hip-to-foot distance in x-axis
 * @param y hip-to-foot distance in y-axis
 * @param z hip-to-foot distance in z-axis
 * @param D Leg domain
 * @param r_o Radius of the leg
 * @param config Simulation environment configuration parameters.
 * 
 * @return Joint Angles required for desired position. 
 *  The order is: Hip, Thigh, Shank
 *  Or: (shoulder, elbow, wrist)
 */
Eigen::Vector3d rightLegIK(
    double x, 
    double y, 
    double z, 
    double D, 
    double r_o,
    EnvConfig *config
) { 
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle    = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = - std::atan2(z, y) - std::atan2(r_o, - config->H_OFF);
    second_sqrt_component = std::max(
        0.0,
        (
            std::pow(r_o, 2) + std::pow(x, 2) - 
            std::pow((config->SHANK_LEN * std::sin(wrist_angle)), 2)
        )
    );
    q_o = std::sqrt(second_sqrt_component);
    elbow_angle = std::atan2(-x, r_o);
    elbow_angle -= std::atan2(config->SHANK_LEN * std::sin(wrist_angle), q_o);

    Eigen::Vector3d joint_angles(-shoulder_angle, elbow_angle, wrist_angle);
    return joint_angles;
}

/**
 * @brief Left Leg Inverse Kinematics Solver
 * 
 * @param x hip-to-foot distance in x-axis
 * @param y hip-to-foot distance in y-axis
 * @param z hip-to-foot distance in z-axis
 * @param D Leg domain
 * @param r_o Radius of the leg
 * @param config Simulation environment configuration parameters.
 * 
 * @return Joint Angles required for desired position. 
 *  The order is: Hip, Thigh, Shank
 *  Or: (shoulder, elbow, wrist)
 */
Eigen::Vector3d leftLegIK(
    double x, 
    double y, 
    double z, 
    double D, 
    double r_o,
    EnvConfig *config
) { 
    // Declare the variables
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle    = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = - std::atan2(z, y) - std::atan2(r_o, config->H_OFF);
    second_sqrt_component = std::max(
        0.0,
        (
            std::pow(r_o, 2) + std::pow(x, 2) - 
            std::pow((config->SHANK_LEN * std::sin(wrist_angle)), 2)
        )
    );
    q_o = std::sqrt(second_sqrt_component);
    elbow_angle = std::atan2(-x, r_o);
    elbow_angle -= std::atan2(config->SHANK_LEN * std::sin(wrist_angle), q_o);

    Eigen::Vector3d joint_angles(-shoulder_angle, elbow_angle, wrist_angle);
    return joint_angles;
}

/**
 * @brief Calculates the leg's inverse kinematics (joint angles from xyz 
 * coordinates).
 * 
 * @param right_leg If true, the right leg is solved, otherwise the left leg 
 *      is solved.
 * @param r Objective foot position in the H_i frame. (x,y,z) hip-to-foot 
 *      distances in each dimension
 * @param config Simulation environment configuration parameters.
 * 
 * @return Leg joint angles to reach the objective foot 
 *      position r. In the order:(Hip, Shoulder, Wrist). The joint angles are 
 *      expresed in radians.
 */
Eigen::Vector3d solveLegIK(bool right_leg, Eigen::Vector3d r, EnvConfig *config)
{
    std::pair<double, double> params = IKParams(r(0), r(1), r(2), config);

    double D   = params.first;
    double r_o = params.second;

    return right_leg ? 
        rightLegIK(r(0), r(1), r(2), D, r_o, config) : 
        leftLegIK(r(0), r(1), r(2), D, r_o, config);
}
