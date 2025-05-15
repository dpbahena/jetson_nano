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



struct Settings {
    int numberOfParticles;
    float particleRadius;
    float spacing;
    float convRadius;
    float alpha;
    

    bool operator!=(const Settings& other) const {
        return std::tie(numberOfParticles, particleRadius, spacing, convRadius, alpha) !=
               std::tie(other.numberOfParticles, other.particleRadius, other.spacing, other.convRadius, other.alpha);
    }
};



class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();
        Lenia* lenia = nullptr;
        std::vector<uchar4> colorPallete = {DARK, BLUE, GREEN, GOLD, WHITE, 
                                            PINK, ORANGE, TAN, BLUE_PLANET, GRAY_ROCKY, SUN_YELLOW, JUPITER, 
                                            SPACE_NIGH, FULL_MOON, RED_MERCURY,  VENUS_TAN, RED_MERCURY, 
                                            MARS_RED, SATURN_ROSE, NEPTUNE_PURPLE, URANUS_BLUE, 
                                            PLUTO_TAN, LITE_GREY };
        // Device Variables
        Particle* d_leniaParticles = nullptr;
        curandState_t* d_states = nullptr;
        uchar4* d_colors = nullptr;

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisplay(cudaSurfaceObject_t &surface, uchar4 color);
        void drawTriangle(cudaSurfaceObject_t &surface, uchar4 colorcolor, vec2 v0, vec2 v1, vec2 v2);

        // program variables
        float dt;  // delta time
        int height, width;
        float zoom = 1.f, panX = 0.f, panY = 0.f;
        bool startSimulation = false;
        int leniaSize = 0;
        int totalParticles = 1e6;
        float particleRadius = .5f;
        float spacing = 1.0f;
        float convolutionRadius = 8.0f;
        float alpha = 4.0;
        float sigma = 0.03f;
        float mu = 0.16f;
        float conv_dt = 0.05f;



        void initLenia();
        std::vector<float> generateCircularShellKernel(int radius, float alpha=4.0f);
        

    private:
        // Cuda kernel configuration variables
        int blockSize; 
        int gridSize;
        
    
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};