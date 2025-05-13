#include "CUDAHandler.h"
#include "cuda_utils.h"




__global__ void clearSurface_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

__device__
void drawPixel(cudaSurfaceObject_t surface, int x, int y, uchar4 color, int width, int height)
{
    if (x >= 0 && x < width && y >= 0 && y < height) {
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }
}

__device__ void drawLine(cudaSurfaceObject_t surface, int x0, int y0, int x1, int y1, uchar4 color, int width, int height)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        drawPixel(surface, x0, y0, color, width, height);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

__device__ void drawCircleOutline(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
    const int segments = 36; // More segments = smoother circle
    for (int i = 0; i < segments; ++i) {
        float theta0 = (2.0f * M_PI * i) / segments;
        float theta1 = (2.0f * M_PI * (i + 1)) / segments;
        
        int x0 = cx + radius * cosf(theta0);
        int y0 = cy + radius * sinf(theta0);
        int x1 = cx + radius * cosf(theta1);
        int y1 = cy + radius * sinf(theta1);

        drawLine(surface, x0, y0, x1, y1, color, width, height);
    }
}

__device__ void drawFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
    int rSquared = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;

        for (int dx = -radius; dx <= radius; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;

            if (dx * dx + dy * dy <= rSquared) {
                drawPixel(surface, x, y, color, width, height);
            }
        }
    }
}

__device__ bool threeEdgeTest2D(const Vec2 &A, const Vec2 &B, const Vec2 &C, const Vec2 &P)
{
    Vec2 a = B - A;
    Vec2 b = C - A;
    Vec2 q = P - A;

    float cross_ab = a.x * b.y - a.y * b.x;

    if (cross_ab == 0.0f) return false; // Triangle is degenerate (no area)

    float s = (q.x * b.y - q.y * b.x) / cross_ab;
    float r = (a.x * q.y - a.y * q.x) / cross_ab;

    return s >= 0.0f && r >= 0.0f && (s + r) <= 1.0f;
}

__device__ __host__ void swap(Vec2& a, Vec2& b) {
        Vec2 temp = a;
        a = b;
        b = temp;
}


__global__ void drawTriangle_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color, int minX, int minY, Vec2 a, Vec2 b, Vec2 c)
{
    int local_x = threadIdx.x + blockIdx.x * blockDim.x;
    int local_y = threadIdx.y + blockIdx.y * blockDim.y;

    int x = local_x + minX;
    int y = local_y + minY;

    if (x >= width || y >= height) return;

    if (!threeEdgeTest2D(a, b, c, Vec2(x, y))) return;
    drawPixel(surface, x, y, color, width, height);

    
}

CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler::CUDAHandler(int width, int height, GLuint textureID) :  width(width), height(height)
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    instance = this; // store global reference (to be used for mouse and imGui User Interface (UI) operations)
}

CUDAHandler::~CUDAHandler()
{
    
    cudaGraphicsUnregisterResource(cudaResource);
    
}
// _________________________________________________________________________//
void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;
 
    cudaSurfaceObject_t surface = MapSurfaceResouse();    
   
    clearGraphicsDisply(surface, PINK);
    drawTriangle(surface, RED_MERCURY, Vec2(100, 200), Vec2(150, 600), Vec2(500, 300));


    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}

//________________________________________________________________________//

void CUDAHandler::clearGraphicsDisply(cudaSurfaceObject_t &surface, uchar4 color)
{
    int threads = 16; 
    dim3 clearBlock(threads, threads);
    dim3 clearGrid((width + clearBlock.x -1) / clearBlock.x, (height + clearBlock.y - 1) / clearBlock.y);
    clearSurface_kernel<<<clearGrid, clearBlock>>>(surface, width, height, color);
}

void CUDAHandler::drawTriangle(cudaSurfaceObject_t &surface, uchar4 color, Vec2 v0, Vec2 v1, Vec2 v2)
{

    // Sort vertices by y-coordinate
    if (v0.y > v1.y) { swap(v0, v1); }
    if (v0.y > v2.y) { swap(v0, v2); }
    if (v1.y > v2.y) { swap(v1, v2); }

    double yMax = v2.y;
    double yMin = v0.y;

    if (v0.x > v1.x) { swap(v0, v1);}
    if (v0.x > v2.x) { swap(v0, v2);}
    if (v1.x > v2.x) { swap(v1, v2);}

    double xMax = v2.x;
    double xMin = v0.x;

    // Calculate bounding box dimensions
    int boxWidth = ceilf(xMax) - floorf(xMin);
    int boxHeight = ceilf(yMax) - floorf(yMin);
    xMin = floorf(xMin);
    yMin = floorf(yMin);

    dim3 threads (16, 16);
    dim3 blocks((boxWidth + threads.x - 1)/ threads.x, (boxHeight + threads.y - 1)/ threads.y);
    drawTriangle_kernel<<<blocks, threads>>>(surface, width, height, RED_MERCURY, xMin, yMin, v0, v1, v2);


}

cudaSurfaceObject_t CUDAHandler::MapSurfaceResouse()
{
    //* Map the resource for CUDA
    cudaArray_t array;
    // glFinish();
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

    //* Create a CUDA surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);
    return surface;
}
