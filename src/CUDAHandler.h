#pragma once


#include "my_vectors.h"
#include "cuda_utils.h"
#include "lenia.h"
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#include <GL/glut.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>



class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();
        Lenia* lenia;
        std::vector<uchar4> colorPallete = {BLUE_PLANET, GRAY_ROCKY, SUN_YELLOW, JUPITER, 
                                            SPACE_NIGH, FULL_MOON, RED_MERCURY,  VENUS_TAN, RED_MERCURY, 
                                            MARS_RED, SATURN_ROSE, NEPTUNE_PURPLE, URANUS_BLUE, 
                                            PLUTO_TAN, LITE_GREY, DARK, BLUE, GREEN, GOLD, WHITE, 
                                            PINK, ORANGE, TAN };
        // Device Variables
        Particle* d_leniaParticles;
        curandState_t* d_states;
        uchar4* d_colors;

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisplay(cudaSurfaceObject_t &surface, uchar4 color);
        void drawTriangle(cudaSurfaceObject_t &surface, uchar4 colorcolor, vec2 v0, vec2 v1, vec2 v2);

        // program variables
        float dt;  // delta time
        int height, width;
        bool startSimulation = false;
        int leniaSize = 0;
        int totalParticles = 1e6;
        float particleRadius = .5f;
        float spacing = 1.0f;

        void initLenia();

    private:
        int blockSize;
        int gridSize;
    
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};