#pragma once


#include "vectors.h"
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>



class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisply(cudaSurfaceObject_t &surface, uchar4 color);
        void drawTriangle(cudaSurfaceObject_t &surface, uchar4 colorcolor, Vec2 v0, Vec2 v1, Vec2 v2);

        // program variables
        float dt;  // delta time
        int height, width;
      
    
       



     

  


      
    
    private:
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};