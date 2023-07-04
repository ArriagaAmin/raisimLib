#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include "EnvConfig.hpp"

/**
 * @brief Quaternion struct. The quaternion is expressed in the form
 *     `w + xi + yj + zk`.
 * 
 * @param w Real part of the quaternion.
 * @param x First imaginary part of the quaternion.
 * @param y Second imaginary part of the quaternion.
 * @param z Third imaginary part of the quaternion.
 * 
*/
struct Quaternion
{
    double w, x, y, z;
};

/**
 * @brief Euler angle struct. The angles are expressed in radians.
 * 
 * @param roll Rotation around the x axis.
 * @param pitch Rotation around the y axis.
 * @param yaw Rotation around the z axis.
*/
struct EulerAngles
{
    double roll, pitch, yaw;
};

/**
 * @brief Computes the euler angle representation from the quaternion
 * representation
 *
 * @param q Orientation Quaternion (Unit quaternion)
 *
 * @return EulerAngles
 */
EulerAngles to_euler_angles(Quaternion q);


/**
 * @brief Computes the quaternion representation from the euler angles
 *
 * @param q EulerAngles
 *
 * @return Quaternion
 */
Quaternion to_quaternion(EulerAngles q);


/** 
 * @brief Computes the rotation matrix from the euler angles representation
 * 
 * @param euler EulerAngles
 * 
 * @return Eigen::Matrix3d
 */

Eigen::Matrix3d euler_to_rotation_matrix(EulerAngles euler);

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
Eigen::Vector3d solve_leg_IK(bool right_leg, Eigen::Vector3d r, EnvConfig *config);

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