#pragma once

#include "my_vectors.h"
#include <vector>

struct Particle {
    vec2 position;
    float radius;
    bool alive;
    bool nextAlive;
    float energy=0.f;
    float nextEnergy=0.f;
    uchar4 color;
};

class Lenia {
    public:
        Lenia();
        Lenia(int displayWidth, int displayHeight, int totalParticles, float particleRadius, float spacing, int2 gridRatio);
        ~Lenia();
        std::vector<Particle> particles;
        vec2 topLeft;
        int gridRows;
        int gridCols;

    private:
        int totalParticles;
        float particleRadius;
        float spacing;
        int width, height;
   

        int2 calculateGridDimensions(int x, int y);
        void setGrid(int2 ratio);
        


};

