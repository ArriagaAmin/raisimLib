#pragma once

/**
 * @brief Simulation environment configuration parameters
 */
struct EnvConfig
{
    // Cartesian foot placement heuristic enabled/disabled
    bool CARTESIAN_DELTA;

    // Angular foot placement heuristic enabled/disabled
    bool ANGULAR_DELTA;

    // Number of scans for each ring.
    int SCANS_PER_RING;

    // Number of rings.
    int N_SCAN_RINGS;

    // Innermost ring radius.
    double FOOT_SCAN_RADIUS;

    // Initial phase of the robot legs
    double SIGMA_0[4];

    // Gait base frequency.
    double BASE_FREQUENCY;

    // Leg span.
    double H;

    // Horizontal offset of the leg.
    double H_OFF;

    // Vertical offset of the leg.
    double V_OFF;

    // Robot thigh length.
    double THIGH_LEN;

    // Robot shank length.
    double SHANK_LEN;

    double LEG_SPAN;

    // Differential time.
    double CONTROL_DT;

    // Desired minimum speed of the robot.
    double VEL_TH;

    // Maximum magnitude of the external force applied to the robot.
    double MAX_EXTERNAL_FORCE;

    // Time during which the external force will be applied to the robot.
    double EXTERNAL_FORCE_TIME;
};
