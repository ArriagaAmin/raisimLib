#include "WorldGenerator.hpp"

WorldGenerator::WorldGenerator(
    raisim::World *world,
    raisim::ArticulatedSystem *anymal) : world_(world), anymal_(anymal){};

void WorldGenerator::step(
    double x,
    double y,
    double lenx,
    double leny,
    double height)
{
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
    int n)
{
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
            step(x - (i + 1) * lenx, y, lenx, leny, (i + 1) * height);
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
    double total_width)
{
    // int n = int(total_length / width);

    // raisim::Ground *ground = this->world_->addGround();

    // this->terrain_objects.push(ground);
    // this->terrain_x_size = total_length;
    // this->terrain_y_size = total_width;

    // stair(0, -total_length / 2, 'E', total_width, width, height, n);
    int resolution = 512;
    this->terrain_x_size = total_width; //total_length;
    this->terrain_y_size = total_length;


    std::vector<double> height_map(resolution * resolution);

    int width_int = std::max(1, (int)(resolution * width / this->terrain_y_size));
    double step_height = 0.0;

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            height_map[i * resolution + j] = step_height;
        }

        // Check if the width is reached
        if (i % width_int == 0)
        {
            step_height = height * (i / width_int + 1);
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
        "ground");
    this->terrain_objects.push(map);
};




void WorldGenerator::hills(
    double frequency,
    double amplitude,
    double roughness,
    double terrain_size,
    double roughness_resolution,
    int resolution)
{
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
        "ground");
    this->terrain_objects.push(map);
};

void WorldGenerator::cellular_steps(
    double frequency,
    double amplitude,
    double terrain_size,
    int resolution)
{
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    noise.SetFractalType(FastNoiseLite::FractalType_Ridged);
    noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
    noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_CellValue);
    noise.SetFrequency(float(frequency));
    noise.SetSeed(rand());

    // Gather noise data
    std::vector<double> height_map(resolution * resolution);
    for (int y = 0; y < resolution; y++)
    {
        for (int x = 0; x < resolution; x++)
        {
            height_map[x * resolution + y] = noise.GetNoise((double)x, (double)y) * amplitude;
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
        "ground");
    this->terrain_objects.push(map);
}



void WorldGenerator::steps(
    double width,
    double height,
    double terrain_size,
    int resolution)
{
    this->terrain_x_size = terrain_size;
    this->terrain_y_size = terrain_size;

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFractalOctaves(2);
    noise.SetFractalLacunarity(30.0);
    noise.SetFrequency(0.01f);
    noise.SetSeed(rand());

    std::vector<double> height_map(resolution * resolution);

    // Create a uniform random distribution between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    int width_int = std::max(1, (int)(resolution * width / this->terrain_x_size));
    double step_height;

    for (int y = 0; y < resolution; y += width_int)
    {
        for (int x = 0; x < resolution; x += width_int)
        {
            step_height = height * noise.GetNoise((double)x, (double)y) * dis(gen);

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
        "ground");
    this->terrain_objects.push(map);
}

void WorldGenerator::slope(
    double slope,
    double roughness,
    double terrain_size,
    int resolution)
{
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
            height_map[y * resolution + x] = roughness * dis(gen) + x * this->terrain_x_size * slope / resolution;
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
        "ground");
    this->terrain_objects.push(map);
}

void WorldGenerator::clear(void)
{
    while (!this->terrain_objects.empty())
    {
        this->world_->removeObject(this->terrain_objects.top());
        this->terrain_objects.pop();
    }
}
