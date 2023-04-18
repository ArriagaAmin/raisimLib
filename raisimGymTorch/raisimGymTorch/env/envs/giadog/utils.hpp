#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include "EnvConfig.hpp"

struct Quaternion
{
    double w, x, y, z;
};

struct EulerAngles
{
    double roll, pitch, yaw;
};

/**
 * @brief Computes the euler angle representation from the quaternion
 * representation
 *
 * @param q Quaternion representation
 *
 * @return EulerAngles
 */
EulerAngles to_euler_angles(Quaternion q);

/**
 * @brief Function used to apply the control pipeline to the robot. The control
 * pipeline is composed of the following steps:
 *      1. Calculate the FTG state for each foot.
 *      2. Sum the residuals to the FTG output foot positions
 *      3. Calculate the heuristic deltas for each foot according to the
 *         desired turn direction and command direction. (Only if desired)
 *      4. Transform the foot positions to the hip frames
 *      5. Apply the inverse kinematics to the obtained foot positions.
 *
 * @param action The action to apply to the robot. The action is a vector of
 *      size 16. The last 4 positions are the frecuencies offsets of each legs
 * @param turn_dir Robot rotation direction.
 * @param command_dir Robot command direction.
 * @param roll Roll axis of the robot base.
 * @param pitch Pitch axis of the robot base.
 * @param time Current time in simulation.
 * @param config Simulation environment configuration parameters.
 *
 * @return Joint angles.
 * @return Feet target positions.
 * @return Foot Trajectory Generator frequencies.
 * @return Sine of the Foot Trajectory Generator phases.
 * @return Cosine of the Foot Trajectory Generator phases.
 * @return Foot Trajectory Generator phases.
 */
std::tuple<
    Eigen::VectorXd,
    Eigen::VectorXd,
    Eigen::Vector4d,
    Eigen::Vector4d,
    Eigen::Vector4d,
    Eigen::Vector4d>
control_pipeline(
    Eigen::VectorXd action,
    int turn_dir,
    double command_dir,
    double roll,
    double pitch,
    double time,
    EnvConfig *config);

/**
 * @brief Given an initial epoch `E` and a duration in epochs `D`, it returns
 * an approximation of the parameters `B`, `d` so that the recursion
 * `an = an-1 ^ d, a0 = B`, which can be expressed as the function
 * `f(n) = B ^ d ^ n` where `n` its the epoch, have a curve that begins to 
 * grow significantly at epoch `E` and its growth spurt lasts `D` epochs.
 *
 * @param desired_epoch_init Epoch at which you want the recursion to start
 * growing significantly
 * @param desired_epoch_len Duration in epochs of the significant growth stage
 * of the recursion
 *
 * @return std::pair<double, double> Pair of approximate parameters `B` and `d`
 * de la recursion `an = an ^ d, a0 = B`.
 */
std::pair<double, double> find_begin_and_decay(
    int desired_epoch_init,
    int desired_epoch_len);