# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import os
import platform
import numpy as np
from raisimGymTorch.env.bin import giadog


class RaisimGymVecEnv:
    """ 
        Class that represents a vector of simultaneous simulations
    """

    def __init__(self, wrapper: giadog, normalize: bool=True):
        """
            Parameters:
                * wrapper: Connection to C++ environments
                * normalize: Indicates whether the observations should be 
                normalized
        """
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.normalize = normalize
        self.wrapper = wrapper

        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros(
            [self.num_envs, self.num_obs], dtype=np.float32)
        self.base_euler_angles = np.zeros([self.num_envs, 3], dtype=np.float32)
        self.actions = np.zeros(
            [self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def hills(self, frequency: float, amplitude: float, roughness: float):
        self.wrapper.hills(frequency, amplitude, roughness)

    def cellular_steps(self, frequency: float, amplitude: float):
        self.wrapper.cellularSteps(frequency, amplitude)

    def stairs(self, width: float, height: float):
        self.wrapper.stairs(width, height)

    def steps(self, width: float, height: float):
        self.wrapper.steps(width, height)

    def slope(self, slope: float, roughness: float):
        self.wrapper.slope(slope, roughness)

    def getTraversability(self):
        trav = self.wrapper.getTraversability()
        return trav

    def getSpeed(self):
        speed = self.wrapper.getSpeed()
        return speed

    def get_projected_speed(self):
        speed = self.wrapper.getProjSpeed()
        return speed

    def get_max_torque(self):
        return self.wrapper.getMaxTorque()

    def get_froude(self):
        return self.wrapper.getFroude()

    def get_power(self):
        return self.wrapper.getPower()

    def get_base_euler_angles(self):
        self.wrapper.get_base_euler_angles(self.base_euler_angles)
        return self.base_euler_angles

    def set_command(self, direction_angle: float, turning_direction: float, stop: bool):
        self.wrapper.setCommand(direction_angle, turning_direction, stop)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
