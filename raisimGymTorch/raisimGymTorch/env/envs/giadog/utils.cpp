#include "utils.hpp"

#define BEGIN 0.1
#define DECAY 0.5
#define MAX_EPOCHS 10000

EulerAngles to_euler_angles(Quaternion q)
{
    EulerAngles angles;

    // Roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
    {
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    }
    else
    {
        angles.pitch = std::asin(sinp);
    }

    // Yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}


Quaternion to_quaternion(EulerAngles euler){
    
    double cr = cos(euler.roll * 0.5);
    double sr = sin(euler.roll * 0.5);

    double cp = cos(euler.pitch * 0.5);
    double sp = sin(euler.pitch * 0.5);

    double cy = cos(euler.yaw * 0.5);
    double sy = sin(euler.yaw * 0.5);
    
    Quaternion q;

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

Eigen::Matrix3d euler_to_rotation_matrix(EulerAngles euler){
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = Eigen::AngleAxisd(euler.roll, Eigen::Vector3d::UnitX()) *
                      Eigen::AngleAxisd(euler.pitch, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(euler.yaw, Eigen::Vector3d::UnitZ());
    return rotation_matrix;
}



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
Eigen::MatrixXd direction_deltas(
    Eigen::Vector4d ftg_freqs,
    Eigen::Vector4d ftg_sine_phases,
    double command_dir,
    int turn_dir,
    bool add_cartesian_delta,
    bool add_angular_delta,
    EnvConfig *config)
{
    double H = config->H; // Gaits foot vertical span
    double dt = config->CONTROL_DT;     // Control time step
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(4, 3);

    for (int i = 0; i < 4; i++)
    {
        Eigen::Vector3d position_delta(0.0, 0.0, 0.0);

        // Position delta
        position_delta(0) = config->X_MOV_DELTA * cos(command_dir) * dt * ftg_sine_phases(i) *
                             H;
        position_delta(1) = config->Y_MOV_DELTA * sin(command_dir) * dt * ftg_sine_phases(i) *
                             H;

        // Rotation delta
        Eigen::Vector3d rotation_delta(0.0, 0.0, 0.0);

        double theta = M_PI / 4;
        double phi_arc = (i == 0) * -theta + (i == 1) * -(M_PI - theta) +
                         (i == 2) * theta + (i == 3) * (M_PI - theta);

        rotation_delta(0) = config->ANG_MOV_DELTA * -cos(phi_arc) * dt * ftg_sine_phases(i) *
                            turn_dir *  H;
        rotation_delta(1) = config->ANG_MOV_DELTA * -sin(phi_arc) * dt * ftg_sine_phases(i) *
                            turn_dir *  H;

        delta.row(i) += (position_delta * add_cartesian_delta +
                         rotation_delta * add_angular_delta); 
    }
    return delta;
}

/**
 * @brief Foot Trajectory Generator.
 *
 * @param sigma_i_0 Leg base phase.
 * @param t   Elapsed time [s].
 * @param f_i Frequency of the i-th leg.
 * @param H   Foot vertical span. 
 * ---------
 *     o -- Max height
 *     |        |
 *     |        |
 *     |        Vertical span
 *     |        |
 *     |        |
 *     o -- Min height (ie floor)
 *
 * @return Foot Z position.
 * @return Foot phase.
 */
std::pair<double, double> FTG(double sigma_i_0, double t, double f_i, double H)
{
    double position_z = 0.0;
    double sigma_i, k;

    sigma_i = std::fmod(sigma_i_0 + t * (f_i), (2 * M_PI));
    k = 2 * (sigma_i - M_PI) / M_PI;

    bool condition_1 = (k <= 1 && k >= 0);
    bool condition_2 = (k >= 1 && k <= 2);

    position_z += H * (-2 * k * k * k + 3 * k * k) * condition_1;
    position_z += H * (2 * k * k * k - 9 * k * k + 12 * k - 4) * condition_2;

    return {position_z, sigma_i};
}

/**
 * @brief Compute the foot trajectories for the given frequencies, for all
 * the four legs for the current time t.
 *
 * @param t Current time.
 * @param frequencies Vector of the four frequencies offsets of the four legs.
 * @param config Simulation environment configuration parameters.
 *
 * @return Target foot positions,
 * @return Foot frequencies.
 * @return Sine of the phases of the feet.
 * @return Cosine of the phases of the feet.
 * @return Phases of the feet.
 */
std::tuple<
    Eigen::Vector4d,
    Eigen::Vector4d,
    Eigen::Vector4d,
    Eigen::Vector4d,
    Eigen::Vector4d>
foot_trajectories(double t, Eigen::Vector4d frequencies, EnvConfig *config)
{
    // The foot phases are set to a trot gait.
    Eigen::Vector4d FTG_frequencies = Eigen::Vector4d::Zero();
    Eigen::Vector4d FTG_sin_phases = Eigen::Vector4d::Zero();
    Eigen::Vector4d FTG_cos_phases = Eigen::Vector4d::Zero();
    Eigen::Vector4d FTG_phases = Eigen::Vector4d::Zero();
    Eigen::Vector4d target_foot_positions = Eigen::Vector4d::Zero();

    for (int i = 0; i < 4; i++)
    {
        double f_i = frequencies[i] + config->BASE_FREQUENCY;
        std::pair<double, double>
            ftg_result = FTG(config->SIGMA_0[i], t, f_i, config->H);
        target_foot_positions(i) = ftg_result.first;
        double sigma_i = ftg_result.second;
        FTG_frequencies[i] = f_i;
        FTG_sin_phases[i] = std::sin(sigma_i);
        FTG_cos_phases[i] = std::cos(sigma_i);
        FTG_phases[i] = sigma_i;
    }

    return {
        target_foot_positions,
        FTG_frequencies,
        FTG_sin_phases,
        FTG_cos_phases,
        FTG_phases};
}

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
std::pair<double, double> inverse_kinematic_params(
    double x,
    double y,
    double z,
    EnvConfig *config)
{
    double r_o, D, sqrt_component;

    sqrt_component = std::max(
        (double)0.0,
        std::pow(z, 2) + std::pow(y, 2) - std::pow(config->H_OFF, 2));
    r_o = std::sqrt(sqrt_component) - config->V_OFF;
    D = (std::pow(r_o, 2) + std::pow(x, 2) - std::pow(config->SHANK_LEN, 2) -
         std::pow(config->THIGH_LEN, 2)) /
        (2 * config->SHANK_LEN * config->THIGH_LEN);
    D = std::max(std::min(D, 1.0), -1.0);

    return {D, r_o};
}

/**
 * @brief Right Leg Inverse Kinematics Solver
 *
 * @param x hip-to-foot distance in x-axis
 * @param y hip-to-foot distance in y-axis
 * @param z hip-to-foot distance in z-axis
 * @param D Leg domain
 * @param r_o Parameter of the leg's outer radius.
 * @param config Simulation environment configuration parameters.
 *
 * @return Joint Angles required for desired position.
 *  The order is: Hip, Thigh, Shank
 *  Or: (shoulder, elbow, wrist)
 */
Eigen::Vector3d right_leg_IK(
    double x,
    double y,
    double z,
    double D,
    double r_o,
    EnvConfig *config)
{
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = -std::atan2(z, y) - std::atan2(r_o, -config->H_OFF);
    second_sqrt_component = std::max(
        0.0,
        (
            std::pow(r_o, 2) + std::pow(x, 2) -
            std::pow((config->SHANK_LEN * std::sin(wrist_angle)), 2)));
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
Eigen::Vector3d left_leg_IK(
    double x,
    double y,
    double z,
    double D,
    double r_o,
    EnvConfig *config)
{
    // Declare the variables
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = -std::atan2(z, y) - std::atan2(r_o, config->H_OFF);
    second_sqrt_component = std::max(
        0.0,
        (
            std::pow(r_o, 2) + std::pow(x, 2) -
            std::pow((config->SHANK_LEN * std::sin(wrist_angle)), 2)));
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
Eigen::Vector3d solve_leg_IK(bool right_leg, Eigen::Vector3d r, EnvConfig *config)
{
    std::pair<double, double> params = inverse_kinematic_params(r(0), r(1), r(2), config);

    double D = params.first;
    double r_o = params.second;

    return right_leg ? right_leg_IK(r(0), r(1), r(2), D, r_o, config) : left_leg_IK(r(0), r(1), r(2), D, r_o, config);
}

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
    EnvConfig *config)
{
    Eigen::Vector4d frequencies = action.tail(4);
    Eigen::VectorXd xyz_residuals = action.head(12);

    // Calculate the FTG target position
    std::tuple<
        Eigen::Vector4d,
        Eigen::Vector4d,
        Eigen::Vector4d,
        Eigen::Vector4d,
        Eigen::Vector4d>
        FTG_data = foot_trajectories(time, frequencies, config);
    Eigen::Vector4d z_ftg = std::get<0>(FTG_data);
    Eigen::Vector4d FTG_frequencies = std::get<1>(FTG_data);
    Eigen::Vector4d FTG_sin_phases = std::get<2>(FTG_data);
    Eigen::Vector4d FTG_cos_phases = std::get<3>(FTG_data);
    Eigen::Vector4d FTG_phases = std::get<4>(FTG_data);
    Eigen::MatrixXd dir_delta = Eigen::MatrixXd::Zero(4, 3);

    dir_delta = direction_deltas(
        FTG_frequencies,
        FTG_cos_phases,
        command_dir,
        turn_dir,
        config->CARTESIAN_DELTA,
        config->ANGULAR_DELTA,
        config) ;

    Eigen::VectorXd feet_target_positions;
    feet_target_positions.setZero(12);
    Eigen::VectorXd joint_angles;
    joint_angles.setZero(12);

    for (int i = 0; i < 4; i++)
    {
        Eigen::VectorXd foot_delta = dir_delta.row(i);

        double x = foot_delta(0) + xyz_residuals(i * 3);
        double y = foot_delta(1) + xyz_residuals(i * 3 + 1) ;
        double z = z_ftg(i) + foot_delta(2) + xyz_residuals(i * 3 + 2);

        feet_target_positions(i * 3) = x;
        feet_target_positions(i * 3 + 1) = y ;
        feet_target_positions(i * 3 + 2) = z ;

        Eigen::Vector3d r;
        if (config->USE_HORIZONTAL_FRAME){
            roll = pitch = 0.0;
            x += 0;
            y += config->H_OFF * pow(-1, i);
            z += -config->LEG_SPAN * (1 - 0.225); 
        }

        //roll = pitch = 0.0;
        // Transform the feet target position to the base horizontal frame
        r = {
            x * std::cos(pitch) + y * std::sin(pitch) * std::sin(roll) + z * std::sin(pitch) * std::cos(roll) + 0,
            0 + y * std::cos(roll) - z * std::sin(roll),
            -x * std::sin(pitch) + y * std::cos(pitch) * std::sin(roll) + z * std::cos(pitch) * std::cos(roll)};
        
        if (!config->USE_HORIZONTAL_FRAME){
            r(1) += config->H_OFF * pow(-1, i);
            r(2) += -config->LEG_SPAN * (1 - 0.225); 
        }

        if ( (i == 2 || i == 3) && config->ROBOT_LEG_CONFIG == "><"){
            r(0) = -r(0);
        }

        bool right_leg = i == 1 || i == 3;
  
        // Asing the joint angles to the joint angle vector.
        auto leg_joint_angles = solve_leg_IK(right_leg, r, config);

        if ( (i == 2 || i == 3) && config->ROBOT_LEG_CONFIG == "><"){
            joint_angles(i * 3)     = leg_joint_angles[0];
            joint_angles(i * 3 + 1) = -leg_joint_angles[1];
            joint_angles(i * 3 + 2) = -leg_joint_angles[2];
        }
        else{  
            joint_angles(i * 3)     = leg_joint_angles[0];
            joint_angles(i * 3 + 1) = leg_joint_angles[1];
            joint_angles(i * 3 + 2) = leg_joint_angles[2];
        }
        
    }

    return std::make_tuple(
        joint_angles,
        feet_target_positions,
        FTG_frequencies,
        FTG_sin_phases,
        FTG_cos_phases,
        FTG_phases);
}

/**
 * @brief Given the parameters `B` and `d`, returns the epoch `n` for which 
 * the recursion `an = an-1 ^ d, a0 = B` begins to grow significantly, and the 
 * range of epochs in which its growth it is significant.
 * 
 * @param begin Parameter `B` of recursion `an = an-1 ^ d, a0 = B`
 * @param decay Parameter `d` of recursion `an = an-1 ^ d, a0 = B`
 * 
 * @return std::pair<int, int> Pair of values epoch `n` for which the 
 * recursion begins to grow significantly, and the range of epochs in which 
 * its growth it is significant.
 */
std::pair<int, int> get_epochs_init_and_len(double begin, double decay)
{
    double max_n = log(-1 / log(begin)) / log(decay);
    double max_v = pow(begin, pow(decay, max_n)) * pow(decay, max_n);
    bool status = false;
    int epoch_init = 0, epoch_end = 0;
    for (int i = 0; i < 10000; ++i)
    {
        double y = pow(begin, pow(decay, i)) * pow(decay, i) / max_v;
        if (y > 0.05 && !status)
        {
            status = true;
            epoch_init = i;
        }
        if (y < 0.05 && status)
        {
            epoch_end = i;
            break;
        }
    }
    return std::make_pair(epoch_init, epoch_end - epoch_init);
}

/**
 * @brief Finds the approximate parameter `d` for which the recursion 
 * `an = an-1 ^d, a0 = B` has a specified range of growing epochs.
 * 
 * @param desired_epoch_len Desired range of growing epochs
 * @return double Parameter `d`
 */
double find_decay(int desired_epoch_len) {
    int current_epoch_len = 0;
    double decay = DECAY;
    while (current_epoch_len < desired_epoch_len) {
        decay += 0.0001;
        current_epoch_len = get_epochs_init_and_len(BEGIN, decay).second;
    }
    return decay;
}

/**
 * @brief Finds the approximate parameter `B` for which the recursion 
 * `an = an-1 ^d, a0 = B` has a specified epoch for the start of its growth.
 * 
 * @param desired_epoch_inir Desired initial epoch of growth
 * @param decay Parameter `d` for which the recursion `an = an-1 ^d, a0 = B`
 * 
 * @return double Parameter `d`
 */
double find_begin(int desired_epoch_init, double decay) {
    int current_epoch_init = 0;
    double begin = BEGIN;
    while (current_epoch_init < desired_epoch_init && begin > 0) {
        begin /= 10;
        current_epoch_init = get_epochs_init_and_len(begin, decay).first;
    }
    return std::max(begin, 1e-100);
}

std::pair<double, double> find_begin_and_decay(int desired_epoch_init, int desired_epoch_len)
{
    double decay = find_decay(desired_epoch_len);
    double begin = find_begin(desired_epoch_init, decay);
    return std::make_pair(begin, decay);
}