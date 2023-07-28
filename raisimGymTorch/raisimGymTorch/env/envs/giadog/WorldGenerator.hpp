#pragma once

#include <stack>
#include "RaisimGymEnv.hpp"
#include "FastNoiseLite.hpp"

/**
 * @brief Class that is used to build the different types of terrain in the
 * simulation environment. The lengths used in the class are based on meters.
 *
 */
class WorldGenerator
{
private:
    // Environment where the simulation occurs.
    raisim::World *world_;
    // Robot that is in the simulation.
    raisim::ArticulatedSystem *anymal_;
    // Objects that this class has inserted into the simulation environment.
    std::stack<raisim::Object *> terrain_objects;

    /**
     * @brief Create a cube inside the terrain
     *
     * @param x Position in x of the cube.
     * @param y Position in y of the cube.
     * @param lenx Length in x of the cube.
     * @param leny Length in y of the cube
     * @param height Height of the cube.
     */
    void step(double x, double y, double lenx, double leny, double height);

    /**
     * @brief Create a stair inside the terrain
     *
     * @param x Position in x of the stair.
     * @param y Position in x of the stair.
     * @param orientation Direction in which the stair is going up. It can
     *      take one of the values: 'N', 'S', 'E' or 'W'
     * @param lenx Length in x of each step.
     * @param leny Length in y of each step
     * @param height Height of each step.
     * @param n Number of steps.
     */
    void stair(
        double x,
        double y,
        char orientation,
        double lenx,
        double leny,
        double height,
        int n);

public:
    // Size in x of the terrain.
    double terrain_x_size;
    // Size in y of the terrain.
    double terrain_y_size;

    WorldGenerator(void){};

    WorldGenerator(
        raisim::World *world,
        raisim::ArticulatedSystem *anymal);

    /**
     * @brief Create the training terrain that contains hills.
     *
     * @param frequency How often each hill appears.
     * @param amplitude Height of the hills.
     * @param roughness Terrain roughness.
     * @param terrain_size Size in x and y of the terrain.
     * @param roughness_resolution Roughness resolution.
     * @param resolution Perlin noise resolution.
     */
    void hills(
        double frequency,
        double amplitude,
        double roughness,
        double terrain_size = 8.0,
        double roughness_resolution = 30.0,
        int resolution = 90);

    /**
     * @brief Create the training terrain that contains stairs.
     *
     * @param width Width of each step.
     * @param height Height of each step.
     * @param total_length Total length of the entire terrain.
     * @param total_width Total width of the entire terrain.
     */
    void stairs(
        double width,
        double height,
        double total_length = 6.0,
        double total_width = 6.0);

    /**
     * @brief Create the training terrain that contains stepped terrain.
     *
     * @param frequency Frequency of the cellular noise.
     * @param amplitude Scale to multiply the cellular noise.
     * @param terrain_size Size in x and y of the terrain.
     * @param resolution Perlin noise resolution.
     */
    void cellular_steps(
        double frequency,
        double amplitude,
        double terrain_size = 8.0,
        int resolution = 256);

    /**
     * @brief Create the training terrain that contains stepped terrain.
     *
     * @param width size of each step.
     * @param height Scale of the steps
     * @param terrain_size Size in x and y of the terrain.
     * @param resolution Perlin noise resolution.
     */
    void steps(
        double width,
        double height,
        double terrain_size = 8.0,
        int resolution = 512);

    /**
     * @brief Create the training terrain that contains sloping terrain.
     *
     * @param slope The slope of the terrain.
     * @param roughness Terrain roughness.
     * @param terrain_size Size in x and y of the terrain.
     * @param resolution Perlin noise resolution.
     */
    void slope(
        double slope,
        double roughness,
        double terrain_size = 8.0,
        int resolution = 90);

    /**
     * @brief Remove all objects from the terrain
     */
    void clear(void);
};
