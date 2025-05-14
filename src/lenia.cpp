#include "lenia.h"
#include <limits>

Lenia::Lenia()
{
}

Lenia::Lenia(int displayWidth, int displayHeight, int totalParticles, float particleRadius, float spacing, int2 gridRatio)
    : width(displayWidth), height(displayHeight), totalParticles(totalParticles), particleRadius(particleRadius), spacing(spacing)
{
    setGrid(gridRatio);
}

Lenia::~Lenia()
{
    particles.clear();
}

int2 Lenia::calculateGridDimensions(int a, int b)
{
    double targetRatio = static_cast<double>(a) / b;
    double bestDiff = std::numeric_limits<double>::max();
    int bestRows = 1, bestCols = totalParticles;

    for (int rows = 1; rows <= totalParticles; ++rows) {
        int cols = (totalParticles + rows - 1) / rows; // ceil(n / rows)
        double currentRatio = static_cast<double>(cols) / rows;
        double diff = std::abs(currentRatio - targetRatio);

        if (diff < bestDiff) {
            bestDiff = diff;
            bestRows = rows;
            bestCols = cols;
        }
    }

    return {bestRows, bestCols};
}


void Lenia::setGrid(int2 ratio)
{
    
    // ratio refers to the proportion of length vs width
    int2 grid = calculateGridDimensions(ratio.x, ratio.y);
   
    int rows = grid.x;
    int cols = grid.y;


    gridRows = rows;
    gridCols = cols; 
    
    particles.reserve(gridRows * gridCols);

    float offsetX = (width  - (cols - 1) * spacing) / 2.0f;
    float offsetY = (height - (rows - 1) * spacing) / 2.0f;
    topLeft = vec2(offsetX, offsetY);
    

    // Place particles in a 2D grid at restLength spacing
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float x = topLeft.x + c * spacing;
            float y = topLeft.y + r * spacing;
            Particle p;
            p.position = vec2(x,y);
            p.radius = particleRadius;
            particles.push_back(p);
        }
    }
}
