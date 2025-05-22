#pragma once

#include <GL/glut.h>
// #include <vector>

typedef struct  {
    uint32_t  width;
    uint32_t height;
}Extent2D;


class GLManager {
    public:
        GLManager(int width, int height);
        ~GLManager();

        void initializeGL();
        void render();
        GLuint getTextureID() const;
        // void uploadKernelTexture(const std::vector<unsigned char>& imageData, int width, int height);
        void uploadTexture(const void* data, int width, int height, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);

        Extent2D getExtent() { return extent;}
        
        

    private:
        GLuint textureID;
        Extent2D extent;
        // int width, height;

        void createTexture();

};

