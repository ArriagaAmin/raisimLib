import numpy as np
from typing import List
from .bin.giadog import RaisimGymEnv, Statistics, Step


class RaisimEnv:
    """ 
    Class that represents a vector of simultaneous simulations
    """

    def __init__(self, resource_dir: str, cfg: str, port: int = 4242, normalize: bool = True):
        """
        Parameters:
            * resource_dir: Directory where the resources needed to build the 
            environment are located
            * cfg: Environments configuration path
            * port: Port through which you can connect to the simulation 
            visualizer.
            * normalize: Indicates whether the observations should be normalized
        """
        self.wrapper = RaisimGymEnv(resource_dir, cfg, port, normalize)
        
    def reset(self, epoch: int = 0):
        """
        Restart all environments
        
        Parameters:
            * epoch: Current train epoch
        """
        self.wrapper.reset(epoch)

    def step(self, actions: np.ndarray) -> List[Step]:
        """
        Perform one step in each simulation
         
        Parameters:
            * actions: Action taken in each environment. It must have 
            dimensions [n, m] where n is the number of parallel environments 
            and m the dimension of the action space.
            
        Returns:
            * List[step_t]: Information returned in each environment
        """
        return self.wrapper.step(actions)
    
    def get_statistics(self) -> Statistics:
        """
        Obtains the statistics of the observations
        
        Returns:
            * Statistics: Observation statistics data
        """
        return self.wrapper.get_statistics()

    def set_statistics(self, stats_data: Statistics):
        """
        Set the statistics of the observations
        
        Parameters:
            * stats_data: Observation statistics data
        """
        self.wrapper.set_statistics(stats_data)
        
    def set_command(self, direction_angle: float, turning_direction: float, stop: bool):
        """
        Sets the robot command direction. This method is used when the robot 
        command type is external.
        
        Parameters:
            * direction_angle: Angle to which the robot must move
            * turning_direction: Turning direction: 1 for clockwise, -1 for 
            counter-clockwise and to not rotate.
            * stop: The robot must not move.
        """
        self.wrapper.set_command(direction_angle, turning_direction, stop)

    def hills(self, frequency: float, amplitude: float, roughness: float):
        """
        Create the terrain that contains hills.
        
        Parameters:
            * frequency: How often each hill appears.
            * amplitude: Height of the hills.
            * roughness: Terrain roughness.
        """
        self.wrapper.hills(frequency, amplitude, roughness)

    def cellular_steps(self, frequency: float, amplitude: float):
        """
        Create the terrain that contains stepped terrain
        
        Parameters:
            * frequency: Frequency of the cellular noise
            * amplitude: Scale to multiply the cellular noise
        """
        self.wrapper.cellular_steps(frequency, amplitude)

    def stairs(self, width: float, height: float):
        """
        Create the terrain that contains stairs.
        
        Parameters:
            * width: Width of each step.
            * height: Height of each step.
        """
        self.wrapper.stairs(width, height)

    def steps(self, width: float, height: float):
        """
        Generates a terrain made of steps (little square boxes)
        
        Parameters:
            * width: Width of each of the steps [m]
            * height: Amplitude of the steps[m]
        """
        self.wrapper.steps(width, height)

    def slope(self, slope: float, roughness: float):
        """
        Generates a terrain made of a slope.
        
        Parameters:
            * slope: The slope of the slope [m]
            * roughness: Terrain roughness.
        """
        self.wrapper.slope(slope, roughness)


