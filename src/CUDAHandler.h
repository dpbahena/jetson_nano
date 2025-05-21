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
#include <tuple> // For std::tie
#include <vector>
#include <string>



struct Settings {
    int numberOfParticles;
    float particleRadius;
    float spacing;
    int convRadius;
    float alpha;
    float sigma;
    float mu;
    float m;
    float s;
    

    bool operator!=(const Settings& other) const {
        return std::tie(numberOfParticles, particleRadius, spacing, convRadius, alpha, sigma, mu, m, s) !=
               std::tie(other.numberOfParticles, other.particleRadius, other.spacing, other.convRadius, other.alpha, other.sigma,other.mu, other.m, other.s);
    }
};



class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();
        Lenia* lenia = nullptr;
        std::vector<uchar4> colorPallete = {DARK, BLUE, GREEN, GOLD, WHITE, 
                                            PINK, ORANGE, JUPITER, 
                                            SPACE_NIGH, FULL_MOON, RED_MERCURY,  VENUS_TAN, RED_MERCURY, 
                                            MARS_RED, SATURN_ROSE, NEPTUNE_PURPLE, TAN, BLUE_PLANET, GRAY_ROCKY, SUN_YELLOW, URANUS_BLUE, 
                                            PLUTO_TAN, LITE_GREY };
        // Device Variables
        Particle* d_leniaParticles = nullptr;
        curandState_t* d_states = nullptr;
        uchar4* d_colors = nullptr;
        float* d_debugU = nullptr;
        float* d_debugGrowth = nullptr;

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisplay(cudaSurfaceObject_t &surface, uchar4 color);
        void drawTriangle(cudaSurfaceObject_t &surface, uchar4 color, vec2 v0, vec2 v1, vec2 v2);
        void drawCircularKernel(cudaSurfaceObject_t &surface);

        // program variables
        float dt;  // delta time
        int height, width;
        float zoom = 1.f, panX = 0.f, panY = 0.f;
        bool startSimulation = false;
        int leniaSize = 0;
        
        float particleRadius = .5f;
        float spacing = 1.0f;
        int convolutionRadius = 8;
        float alpha = 4.0;
        float sigma = 0.04f;
        float mu = 0.1f;
        float m = .5f;
        float s = .15f;
        float conv_dt = 0.05f;
        int TARGET_FPS = 90;
        #if defined(__aarch64__) || defined(USE_X11_MONITORS)
        int totalParticles = 150000;
        #else
        int totalParticles = 1e6;
        #endif
        // save statistics
        float debugU_host = 0.0f;
        float debugGrowth_host = 0.0f;
        std::vector<float> uHistory, growthHistory;



        void initLenia();
        std::vector<float> generateCircularShellKernel(int radius, float alpha=4.0f);
        std::vector<float> generateCircularBellKernel(int radius, float m, float s);
        

    private:
        // Cuda kernel configuration variables
        int blockSize; 
        int gridSize;
        
    
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResourse();

};