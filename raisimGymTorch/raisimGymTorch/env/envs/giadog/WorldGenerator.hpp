#include <stack> 
#include "RaisimGymEnv.hpp"
#include "FastNoiseLite.hpp"

/**
 * @brief Class that is used to build the different types of terrain in the 
 * simulation environment. The lengths used in the class are based on meters.
 */ 
class WorldGenerator
{
    private:
        // Environment where the simulation occurs.
        raisim::World *world_;
        // Robot that is in the simulation.
        raisim::ArticulatedSystem* anymal_;
        // Objects that this class has inserted into the simulation environment.
        std::stack<raisim::Object*> terrain_objects;

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
            int n
        );

    public:
        // Size in x of the terrain.
        double terrain_x_size;
        // Size in y of the terrain.
        double terrain_y_size;

        explicit WorldGenerator(void) { }

        explicit WorldGenerator(
            raisim::World *world, 
            raisim::ArticulatedSystem* anymal
        ) : world_(world), anymal_(anymal) { };

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
            double terrain_size=4.0,
            double roughness_resolution=30.0,
            int resolution=90
        );

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
            double total_length=15.0,
            double total_width=7.5
        );

        /**
         * @brief Create the training terrain that contains stepped terrain.
         * 
         * @param frequency Frequency of the cellular noise.
         * @param amplitude Scale to multiply the cellular noise.
         * @param terrain_size Size in x and y of the terrain.
         * @param resolution Perlin noise resolution.
         */
        void cellularSteps(
            double frequency, 
            double amplitude, 
            double terrain_size=4.0,
            int resolution=256
        );

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
            double terrain_size=4.0, 
            int resolution=512
        );

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
            double terrain_size=4.0, 
            int resolution=90
        );

        /**
         * @brief Remove all objects from the terrain
         */
        void clear(void);
};


void WorldGenerator::step(
    double x, 
    double y, 
    double lenx, 
    double leny, 
    double height
) {
    // Create the cube
    raisim::Box *box = world_->addBox(lenx, leny, height, 1, "ground");
    box->setBodyType(raisim::BodyType::STATIC);
    box->setPosition(x, y, height / 2);
    box->setAppearance("1,1,1,1");

    this->terrain_objects.push(box);
}

void WorldGenerator::stair(
    double x,
    double y,
    char orientation,
    double lenx,
    double leny,
    double height,
    int n
) {
    for (int i = 0; i < n; i++)
    {
        switch (orientation)
        {
            case 'E':
                step(x, y + i * leny, lenx, leny, (i + 1) * height);
                break;

            case 'S':
                step(x + i * lenx, y, lenx, leny, (i + 1) * height);
                break;

            case 'W':
                step(x, y - (i + 1) * leny, lenx, leny, (i + 1) * height);
                break;

            case 'N':
                step(x - (i + 1)* lenx, y, lenx, leny, (i + 1) * height);
                break;

            default:
                break;
        }
    }
}

void WorldGenerator::stairs(
    double width, 
    double height,
    double total_length,
    double total_width
) {
    int n = total_length / width;
    raisim::Ground *ground = this->world_->addGround();

    this->terrain_objects.push(ground);
    this->terrain_x_size = total_length;
    this->terrain_y_size = total_width;
    
    stair(0, -total_length / 2, 'E', total_width, width, height, n);
};

void WorldGenerator::hills(
    double frequency, 
    double amplitude, 
    double roughness,
    double terrain_size,
    double roughness_resolution,
    int resolution
) {   
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;

    // Use terrain generator to generate the map
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = frequency;
    terrainProperties.zScale = amplitude;
    terrainProperties.fractalGain = roughness;
    terrainProperties.xSize = this->terrain_x_size;
    terrainProperties.ySize = this->terrain_y_size;
    terrainProperties.xSamples = resolution; 
    terrainProperties.ySamples = resolution; 
    terrainProperties.fractalOctaves = 2;
    terrainProperties.fractalLacunarity = roughness_resolution;
    terrainProperties.seed = rand();

    raisim::HeightMap *map = this->world_->addHeightMap(
        0.0, 
        0.0, 
        terrainProperties, 
        "ground"
    );
    this->terrain_objects.push(map);
};


void WorldGenerator::cellularSteps(
    double frequency, 
    double amplitude,
    double terrain_size,
    int resolution
) {   
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    noise.SetFractalType(FastNoiseLite::FractalType_Ridged);
    noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
    noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_CellValue);
    noise.SetFrequency(frequency);
    noise.SetSeed(rand());

    // Gather noise data
    std::vector<double> height_map(resolution * resolution);
    for (int y = 0; y < resolution; y++)
    {
        for (int x = 0; x < resolution; x++)
        {
            height_map[x * resolution + y] = noise.GetNoise(
                (double) x, 
                (double) y
            ) * amplitude ;
        }
    }

    raisim::HeightMap *map = this->world_->addHeightMap(
        resolution, 
        resolution, 
        this->terrain_x_size, 
        this->terrain_y_size, 
        0.0, 
        0.0, 
        height_map, 
        "ground"
    );
    this->terrain_objects.push(map);
}


void WorldGenerator::steps(
    double width, 
    double height,
    double terrain_size,
    int resolution
) {   
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFractalOctaves(2);
    noise.SetFractalLacunarity(30.0);
    noise.SetFrequency(0.01);
    noise.SetSeed(rand());

    std::vector<double> height_map(resolution * resolution);

    // Create a uniform random distribution between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int width_int = std::max(1, (int) (resolution * width / this->terrain_x_size));
    double step_height;

    for (int y = 0; y < resolution; y += width_int)
    {
        for (int x = 0; x < resolution; x += width_int)
        {
            step_height = height * noise.GetNoise(
                (double) x, 
                (double) y
            ) * dis(gen);
            
            for (int i = y; i < std::min(resolution, y + width_int); i++)
            {
                for (int j = x; j < std::min(resolution, x + width_int); j++)
                {
                    height_map[i * resolution + j] = step_height;
                }
            }
        }
    }

    
    raisim::HeightMap *map = this->world_->addHeightMap(
        resolution, 
        resolution, 
        this->terrain_x_size, 
        this->terrain_y_size, 
        0.0, 
        0.0, 
        height_map, 
        "ground"
    );
    this->terrain_objects.push(map);
}


void WorldGenerator::slope(
    double slope, 
    double roughness,
    double terrain_size,
    int resolution
) {   
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;
    std::vector<double> height_map(resolution * resolution);

    // Create a uniform random distribution between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int y = 0; y < resolution; y++)
    {
        for (int x = 0; x < resolution; x++)
        {
            height_map[y * resolution + x] = roughness * dis(gen) + x * \
                this->terrain_x_size * slope / resolution;
        }
    }
    
    raisim::HeightMap *map = this->world_->addHeightMap(
        resolution, 
        resolution, 
        this->terrain_x_size, 
        this->terrain_y_size, 
        0.0, 
        0.0, 
        height_map, 
        "ground"
    );
    this->terrain_objects.push(map);
}

void WorldGenerator::clear(void) {
    while (! this->terrain_objects.empty()) 
    {
        this->world_->removeObject(this->terrain_objects.top());
        this->terrain_objects.pop();
    }
}


