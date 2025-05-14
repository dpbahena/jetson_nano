#pragma once

#include "my_vectors.h"
#include <vector>

struct Particle {
    vec2 position;
    int radius;
    bool alive;
    bool nextAlive;
    float energy=0.f;
    float nextEnergy;
    uchar4 color;
};

class Lenia {
    public:
        Lenia();
        Lenia(int displayWidth, int displayHeight, int totalParticles, float particleRadius, int2 gridRatio);
        ~Lenia();
        std::vector<Particle*> particles;
        vec2 topLeft;
        
        

    private:
        int totalParticles;
        float particleRadius;
        int gridRows;
        int gridCols;
        int spacing = 2.0;
        int width, height;
        float screenRatio;

        int2 calculateGridDimensions(int x, int y);
        void setGrid(int2 ratio);
        


};

