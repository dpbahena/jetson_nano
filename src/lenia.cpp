#include "lenia.h"
#include <limits>

Lenia::Lenia()
{
}

Lenia::Lenia(int displayWidth, int displayHeight, int totalParticles, float particleRadius, int2 gridRatio)
    : width(displayWidth), height(displayHeight), totalParticles(totalParticles), particleRadius(particleRadius)
{

    particles.clear();
    screenRatio = static_cast<float>(height) / width;
    setGrid(gridRatio);
}

Lenia::~Lenia()
{
    for (auto &p : particles)
        delete p;
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

    // printf("Rows: %d  -  Cols: %d - total: %d\n", rows, cols, rows * cols);

    gridRows = rows;
    gridCols = cols;    

    // int offset = width / 2.0f - (cols - 1) * particleRadius;    
    // float offset = width / 2.0f - (cols - 1) * restLength / 2.0f;
    float offsetX = (width  - (cols - 1) * spacing) / 2.0f;
    float offsetY = (height - (rows - 1) * spacing) / 2.0f;
    topLeft = vec2(offsetX, offsetY);


    // topLeft = vec2(offset, top);
    
    int rowsSize = gridRows;
    int colsSize = gridCols * screenRatio;  // screen ratio for correctness

    // Place particles in a 2D grid at restLength spacing
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float x = topLeft.x + c * spacing;
            float y = topLeft.y + r * spacing;
            Particle* p = new Particle();
            p->position = vec2(x,y);
            p->radius = particleRadius;
            particles.push_back(p);
        }
    }
}
