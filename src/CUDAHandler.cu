#include "CUDAHandler.h"
#include <algorithm>


constexpr int MAX_RADIUS = 12;                    // change to suit your max
constexpr int MAX_K      = (2*MAX_RADIUS+1)*(2*MAX_RADIUS+1);

__constant__ float d_kernelConst[MAX_K];          // <── NOT a pointer



__device__ float random_float_in_range(curandState_t* state, float a, float b) {
    // return a + (b - a) * curand_uniform_float(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
    return a + (b - a) * (curand_uniform(state) - 0.5) * 2.0;  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

__device__
void drawPixel(cudaSurfaceObject_t surface, int x, int y, uchar4 color, int width, int height)
{
    if (x >= 0 && x < width && y >= 0 && y < height) {
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
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


__global__ void init_random(unsigned int seed, curandState_t* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}
__global__ void initializeLeniaParticle(Particle* particles, int totalParticles, curandState_t* states, uchar4* colors, int numberColors) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;
    curandState_t x = states[i];
    // float e =  random_float_in_range(&x, 0.1, 0.5);
    // initial noise
    float e = curand_uniform(&x) * 0.5f;
    // float e = curand_uniform(&x) < 0.15f ? curand_uniform(&x) * 0.35f : 0.0f; 
    particles[i].energy = e;
    states[i] = x; // reinstate the random value;
    int colorIndex = static_cast<int>(e * (numberColors - 1));
    colorIndex = max(0, min(colorIndex, numberColors - 1));
    particles[i].color = colors[colorIndex];
}

// __global__ void drawLeniaParticles(cudaSurfaceObject_t surface, Particle* particles, int numberParticles, int width, int height, float zoom, float panX, float panY){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numberParticles) return;
//     Particle p = particles[i];
//     vec2 pos = p.position;
//     // if (threadIdx.x == 0){
//     //     printf("position.x  %f\n", pos.x);
//     //     printf("Energy %f\n", p.energy);
//     // }
//     float radius = p.radius * zoom;
//     int x0 = (int)(width / 2.0f + (pos.x + panX) * zoom);
//     int y0 = (int)(height / 2.0f + (pos.y + panY) * zoom);
    
//     drawFilledCircle(surface, x0, y0, radius, p.color, width, height);
// }

__global__ void commitNextEnergy_kernel(Particle* particles, int totalParticles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;

    particles[i].energy = particles[i].nextEnergy;
    particles[i].nextEnergy = 0;
}

__global__ void thresholdAndCommit_kernel(Particle* g,
                                          int n,
                                          const uchar4* colors,
                                          int numColors,
                                          float thresh=0.2f /* e.g. 0.3f */)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    const float e = g[i].energy;          // already up-to-date from step-2

    // 1.  Boolean life flag (only used if you still need a discrete state)
    g[i].alive = (e > thresh);

    // 2.  Colour lookup for rendering – linear ramp through your palette
    int idx = (int)(e * (numColors - 1) + 0.5f);
    idx = max(0, min(idx, numColors - 1));
    g[i].color = colors[idx];
}

// __device__ float growthMapping(float u, float mu, float sigma) {
//     return 2.0f * expf(-powf((u - mu), 2) / (2.0f * sigma * sigma)) - 1.0f;
// }
__device__ float myGrowthMapping(float u, float mu, float sigma) {
    float z = ((u - mu) * (u - mu)) / ( 2 * sigma * sigma);
    return 2 * expf(-z) - 1.0f;
}
__device__ float growthMapping(float u, float mu, float sigma) {
    float z = (u - mu) / sigma;
    return expf(-0.5f * z * z) * 2.0f - 1.0f;
}
__device__ float growthMappingquad4(float u, float mu, float sigma) {
    float x = 1.0f - ((u - mu) * (u - mu)) / (9.0f * sigma * sigma);
    x = max(0.0f, x);
    return std::pow(x, 4) * 2.0f - 1.0f;
}

__device__ float growth_quad4(float u, float mu, float sigma) {
    float x = 1.0f - ((u - mu) * (u - mu)) / (9.0f * sigma * sigma);
    x = fmaxf(0.0f, x);
    return powf(x, 4) * 2.0f - 1.0f;
}
__device__ float growth_gaussian(float u, float mu, float sigma) {
    float x = (u - mu) / sigma;
    return 2.0f * expf(-x * x * 4.5f) - 1.0f;
}


__device__ float growth_triangle(float u, float mu, float sigma) {
    float dist = fabsf(u - mu);
    float x = 1.0f - dist / (1.5f * sigma);
    return fmaxf(0.0f, x * 2.0f - 1.0f);
}
__device__ float growth_step(float u, float mu, float sigma) {
    return (fabsf(u - mu) < sigma) ? 1.0f : -1.0f;
}
__device__ float growth_relu(float u, float mu, float sigma) {
    float x = 1.0f - fabsf(u - mu) / sigma;
    return fmaxf(0.0f, x) * 2.0f - 1.0f;
}




// __global__ void activate_LeniaGoL_convolution_kernel(
//     Particle* particles,
//     int totalParticles,
//     int gridRows,
//     int gridCols,
//     int kernelDiameter,
//     int radius,
//     float sigma,
//     float mu,
//     float dt
// ) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i >= totalParticles) return;

//     int row = i / gridCols;
//     int col = i % gridCols;

//     float neighborSum = 0.0f;
//     float weightSum = 0.0f;

//     for (int dr = -radius; dr <= radius; ++dr) {
//         for (int dc = -radius; dc <= radius; ++dc) {
//             int nr = row + dr;
//             int nc = col + dc;
//             // float distance = sqrtf(dr * dr + dc * dc);
//             if (nr >= 0 && nr < gridRows && nc >= 0 && nc < gridCols) {
//                 int j = nr * gridCols + nc;
//                 // float weight = gaussianKernel(distance / radius, sigma);
//                 int kernelIndex = (dr + radius) * kernelDiameter + (dc + radius);
//                 float weight = d_kernelConst[kernelIndex];
//                 neighborSum += particles[j].energy * weight;
//                 weightSum += weight;
//             }
//         }
//     }

   
//     float u = neighborSum  / weightSum ;
    

//     float growth = growthMapping(u, mu, sigma);
//     float e = fminf(1.0f, fmaxf(0.0f, particles[i].energy + dt * growth));
//     particles[i].nextEnergy = e;
// }



__global__ void drawLeniaParticles(cudaSurfaceObject_t surface, Particle* particles, int numberParticles, int width, int height, float zoom, float panX, float panY){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberParticles ) return;

    Particle p = particles[i];

    // Apply zoom and pan directly (without adding screen center shift)
    float radius = p.radius * zoom;
    int x0 = (int)((p.position.x + panX) * zoom);
    int y0 = (int)((p.position.y + panY) * zoom);

    // Reject particles that fall outside the surface
    if (x0 < 0 || x0 >= width || y0 < 0 || y0 >= height) return;

    drawFilledCircle(surface, x0, y0, radius, p.color, width, height);
}

__global__ void activate_LeniaGoL_convolution_kernel(
    Particle* particles,
    int totalParticles,
    int gridRows,
    int gridCols,
    int kernelDiameter,
    int radius,
    float sigma,
    float mu,
    float dt 
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;

    int row = i / gridCols;
    int col = i % gridCols;

    float neighborSum = 0.0f;
    float weightSum = 0.0f;

    for (int dr = -radius; dr <= radius; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
            // Wrap around (periodic boundary)
            int nr = (row + dr + gridRows) % gridRows;
            int nc = (col + dc + gridCols) % gridCols;

            int j = nr * gridCols + nc;
            int kernelIndex = (dr + radius) * kernelDiameter + (dc + radius);

            float weight = d_kernelConst[kernelIndex];
            neighborSum += particles[j].energy * weight;
            weightSum += weight;
        }
    }

    // float u = (weightSum > 0.0f) ? (neighborSum / weightSum) : 0.0f;
    float u = neighborSum;
    float growth = growthMapping(u, mu, sigma);
    float e = fminf(1.0f, fmaxf(0.0f, particles[i].energy + dt * growth));
    // TEMP: Visualize normalized excitation and growth directly
    uchar4 debugColor;
    debugColor.x = (unsigned char)(255.0f * fminf(fmaxf(u, 0.0f), 1.0f));        // Red = excitation u
    debugColor.y = (unsigned char)(255.0f * fminf(fmaxf((growth + 1.0f) * 0.5f, 0.0f), 1.0f)); // Green = growth (-1 to 1 mapped to 0-1)
    debugColor.z = 0;
    debugColor.w = 255;

    particles[i].color = debugColor;

    particles[i].nextEnergy = e;
   

}

__global__ void activate_LeniaGoL_convolution_kernel(
    Particle* particles,
    int totalParticles,
    int gridRows,
    int gridCols,
    int kernelDiameter,
    int radius,
    float sigma,
    float mu,
    float dt,
    float* debugU,
    float* debugGrowth 
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;

    int row = i / gridCols;
    int col = i % gridCols;

    float neighborSum = 0.0f;
    float weightSum = 0.0f;

    for (int dr = -radius; dr <= radius; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
            // Wrap around (periodic boundary)
            int nr = (row + dr + gridRows) % gridRows;
            int nc = (col + dc + gridCols) % gridCols;

            int j = nr * gridCols + nc;
            int kernelIndex = (dr + radius) * kernelDiameter + (dc + radius);

            float weight = d_kernelConst[kernelIndex];
            neighborSum += particles[j].energy * weight;
            weightSum += weight;
        }
    }

    float u = (weightSum > 0.0f) ? (neighborSum / weightSum) : 0.0f;

    float growth = myGrowthMapping(u, mu, sigma);
    
    // float growth = growthMappingquad4(u, mu, sigma);
    // float growth = growth_triangle(u, mu, sigma);
    // float growth = growth_gaussian(u, mu, sigma);
    // float growth = growth_relu(u, mu, sigma);
    // float growth = growth_step(u, mu, sigma);
    float e = fminf(1.0f, fmaxf(0.0f, particles[i].energy + dt * growth));
    // TEMP: Visualize normalized excitation and growth directly
    // uchar4 debugColor;
    // debugColor.x = (unsigned char)(255.0f * fminf(fmaxf(u, 0.0f), 1.0f));        // Red = excitation u
    // debugColor.y = (unsigned char)(255.0f * fminf(fmaxf((growth + 1.0f) * 0.5f, 0.0f), 1.0f)); // Green = growth (-1 to 1 mapped to 0-1)
    // debugColor.z = 0;
    // debugColor.w = 255;

    // particles[i].color = debugColor;

    particles[i].nextEnergy = e;
    // if (i == 0) {
    //     // Store a sample (first thread only — for simplicity)
    //     debugU[0] = u;
    //     debugGrowth[0] = growth;
    // }
    if (i == 0) {
        if (!isnan(u) && !isnan(growth)) {
            debugU[0] = u;
            debugGrowth[0] = growth;
        } else {
            debugU[0] = 0.0f;
            debugGrowth[0] = 0.0f;
        }
    }
    // if (i == 0) {
    //     printf("Sample u = %.5f, growth = %.5f\n", u, growth);
    // }


   

}


__global__ void clearSurface_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}



// __device__
// void drawPixel(cudaSurfaceObject_t surface, int x, int y, uchar4 color, int width, int height)
// {
//     if (x >= 0 && x < width && y >= 0 && y < height) {
//         surf2Dwrite(color, surface, x * sizeof(uchar4), y);
//     }
// }

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

// __device__ void drawFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
//     int rSquared = radius * radius;
//     for (int dy = -radius; dy <= radius; ++dy) {
//         int y = cy + dy;
//         if (y < 0 || y >= height) continue;

//         for (int dx = -radius; dx <= radius; ++dx) {
//             int x = cx + dx;
//             if (x < 0 || x >= width) continue;

//             if (dx * dx + dy * dy <= rSquared) {
//                 drawPixel(surface, x, y, color, width, height);
//             }
//         }
//     }
// }

__device__ bool threeEdgeTest2D(const vec2 &A, const vec2 &B, const vec2 &C, const vec2 &P)
{
    vec2 a = B - A;
    vec2 b = C - A;
    vec2 q = P - A;

    float cross_ab = a.x * b.y - a.y * b.x;

    if (cross_ab == 0.0f) return false; // Triangle is degenerate (no area)

    float s = (q.x * b.y - q.y * b.x) / cross_ab;
    float r = (a.x * q.y - a.y * q.x) / cross_ab;

    return s >= 0.0f && r >= 0.0f && (s + r) <= 1.0f;
}

__device__ __host__ void swap(vec2& a, vec2& b) {
        vec2 temp = a;
        a = b;
        b = temp;
}


__global__ void drawTriangle_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color, int minX, int minY, vec2 a, vec2 b, vec2 c)
{
    int local_x = threadIdx.x + blockIdx.x * blockDim.x;
    int local_y = threadIdx.y + blockIdx.y * blockDim.y;

    int x = local_x + minX;
    int y = local_y + minY;

    if (x >= width || y >= height) return;

    if (!threeEdgeTest2D(a, b, c, vec2(x, y))) return;
    drawPixel(surface, x, y, color, width, height);

    
}


// ────────────────────────────────
//  Chan-Lenia “shell” kernel
//  (build once on the host)
// ────────────────────────────────
std::vector<float> CUDAHandler::generateCircularShellKernel(int radius, float alpha /* ≈4 */)
{
    int  diameter = 2 * radius + 1;
    std::vector<float> kernel(diameter * diameter, 0.0f);

    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
        {
            float dist = std::sqrt(float(x*x + y*y));     // Euclidean distance
            if (dist > radius) continue;                  // outside the circle

            float r = dist / radius;                      // normalise to [0,1]
            float value = std::exp(alpha - alpha / (4.0f * r * (1.0f - r)));
            // ────────────────▲  shell(r, alpha)
            // float value = std::exp(-(r * r) / (2.0f * alpha * alpha));
            // float value = std::exp(alpha - (alpha/(4 * r * (1 - r))));
            // float value = std::pow(4 * r * (1 - r), alpha);
            // float value = std::exp(alpha -1.0f / (2.0f * r * r));

            kernel[(y + radius) * diameter + (x + radius)] = value;
            sum += value;
        }

    // normalise so ΣK = 1 (preserves total mass)
    for (float& v : kernel) v /= sum;

    return kernel;
}

std::vector<float> CUDAHandler::generateCircularBellKernel(int radius, float m, float s)
{
    kernelSize = 2 * radius + 1;
    std::vector<float> kernel(kernelSize * kernelSize, 0.0f);

    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            float dist = std::sqrt(float(x * x + y * y)); // Euclidean distance
            if (dist > radius) continue;                 // outside the circle

            float r = dist / radius;                     // normalize to [0, 1]

            // --- Lenia-style bell curve ---
            // float value = 2.0f *  std::exp(-((r - m) * (r - m)) / (2.0f * s * s)) - 1.f;
            float value = std::exp(-((r - m) * (r - m)) / (2.0f * s * s));

            kernel[(y + radius) * kernelSize + (x + radius)] = value;
            sum += value;
        }
    }

    // normalize so that sum(kernel) = 1
    for (float& v : kernel) v /= sum;

    return kernel;
}


CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler::CUDAHandler(int width, int height, GLuint textureID) :  width(width), height(height)
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    instance = this; // store global reference (to be used for mouse and imGui User Interface (UI) operations)
}

CUDAHandler::~CUDAHandler()
{
    cudaFree(d_leniaParticles); 
    cudaFree(d_colors);
    cudaFree(d_states);
    cudaFree(d_debugGrowth);
    cudaFree(d_debugU);
    delete lenia;
      
    cudaGraphicsUnregisterResource(cudaResource);
    
}

// std::vector<float> CUDAHandler::uHistory;
// std::vector<float> CUDAHandler::growthHistory;
// _________________________________________________________________________//
void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;

    static Settings previousSettings;
    Settings currentSettings = {
        .numberOfParticles = totalParticles,
        .particleRadius = particleRadius,
        .spacing = spacing,
        .convRadius = convolutionRadius,
        .alpha = alpha,
        .sigma = sigma,
        .mu = mu,
        .m = m,
        .s = s,
        .conv_dt = conv_dt
        
    };
    if(leniaSize == 0 || currentSettings != previousSettings) {
        
        initLenia();
        previousSettings = currentSettings;
        
    }
    

    if(startSimulation){
        int kernelDiameter = 2 * convolutionRadius + 1;
        activate_LeniaGoL_convolution_kernel<<<gridSize, blockSize>>>(d_leniaParticles, leniaSize, lenia->gridRows, lenia->gridCols, kernelDiameter, convolutionRadius, sigma, mu, conv_dt, d_debugU, d_debugGrowth);
        commitNextEnergy_kernel<<<gridSize, blockSize>>> (d_leniaParticles, leniaSize);
        thresholdAndCommit_kernel<<<gridSize, blockSize>>> (d_leniaParticles, leniaSize, d_colors, colorPallete.size());
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(&debugU_host, d_debugU, sizeof(float), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(&debugGrowth_host, d_debugGrowth, sizeof(float), cudaMemcpyDeviceToHost));
        // Store it in a rolling buffer
        if (uHistory.size() > 100) uHistory.erase(uHistory.begin());
        uHistory.push_back(debugU_host);
     
        if (growthHistory.size() > 100) growthHistory.erase(growthHistory.begin());
        growthHistory.push_back(debugGrowth_host);


           
    }
 
    cudaSurfaceObject_t surface = MapSurfaceResourse();   

    

    clearGraphicsDisplay(surface, WHITE);
    // drawTriangle(surface, RED_MERCURY, vec2(100, 200), vec2(150, 600), vec2(500, 300));
    // checkCuda(cudaPeekAtLastError());
    // checkCuda(cudaDeviceSynchronize());
    // printf("again: %d\n", leniaSize);
    drawLeniaParticles<<<gridSize, blockSize>>>(surface, d_leniaParticles, leniaSize, width, height, 1.0f, 0.0f, 0.0f);

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

   
    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}

//________________________________________________________________________//

void CUDAHandler::clearGraphicsDisplay(cudaSurfaceObject_t &surface, uchar4 color)
{
    int threads = 16; 
    dim3 clearBlock(threads, threads);
    dim3 clearGrid((width + clearBlock.x -1) / clearBlock.x, (height + clearBlock.y - 1) / clearBlock.y);
    clearSurface_kernel<<<clearGrid, clearBlock>>>(surface, width, height, color);
}

void CUDAHandler::drawTriangle(cudaSurfaceObject_t &surface, uchar4 color, vec2 v0, vec2 v1, vec2 v2)
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

void CUDAHandler::drawCircularKernel(cudaSurfaceObject_t &surface)
{

    // Calculate bounding box   // if not zoom , nor panx, nor pany involved
    // vec2 position(300, 300);
    
    // int xMin = max(0, (int)(position.x - convolutionRadius));
    // int xMax = min(width - 1, (int)(position.x + convolutionRadius));
    // int yMin = max(0, (int)(position.y - convolutionRadius));
    // int yMax = min(height - 1, (int)(position.y + convolutionRadius));

    // int drawWidth   = xMax - xMin + 1;
    // int drawHeight  = yMax - yMin + 1;

    // dim3 blockSize(16, 16);
    // dim3 gridSize ((drawWidth + blockSize.x - 1) / blockSize.x, (drawHeight + blockSize.y -1) / blockSize.y);
    // drawGlowingCircle_kernel<<<gridSize, blockSize>>>(surface, width, height, position.x, position.y, convolutionRadius,  color, 1.5f, xmin, ymin, zoom, panX, panY);
}

void CUDAHandler::initLenia()
{
    
    // Free previously allocated GPU memory if it exists
    if (d_leniaParticles) { 
        checkCuda(cudaFree(d_leniaParticles));
        d_leniaParticles = nullptr;
    }
    if (d_states) {
        checkCuda(cudaFree(d_states));
        d_states = nullptr;
    }
    if (d_colors) {
        checkCuda(cudaFree(d_colors));
        d_colors = nullptr;
    }
    if (d_debugGrowth) {
        checkCuda(cudaFree(d_debugGrowth));
        d_debugGrowth = nullptr;
    }
    if (d_debugU) {
        checkCuda(cudaFree(d_debugU));
        d_debugU = nullptr;
    }


    // Optional: delete previous Lenia object
    if (lenia) {
        delete lenia;
        lenia = nullptr;
    }

    // convKernel = generateCircularShellKernel(convolutionRadius, alpha);
    convKernel = generateCircularBellKernel(convolutionRadius, mu, sigma);

     
    
    size_t bytes = convKernel.size() * sizeof(float);
    checkCuda(cudaMemcpyToSymbol(d_kernelConst, convKernel.data(), bytes));
    int kernelSize = 2 * convolutionRadius + 1;
    imageData.resize(kernelSize * kernelSize * 4);

    float minVal = *std::min_element(convKernel.begin(), convKernel.end());
    float maxVal = *std::max_element(convKernel.begin(), convKernel.end());


    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            float value = convKernel[y * kernelSize + x];  // Assume normalized in [0, 1]
            // value = fminf(fmaxf(value, 0.0f), 1.0f);          // Clamp

            // // Map to a color (e.g., blue–yellow gradient)
            // float r = value;
            // float g = value * 0.6f + 0.4f; // emphasize mid values
            // float b = 1.0f - value;

            // int idx = (y * kernelSize + x) * 4;
            // imageData[idx + 0] = static_cast<unsigned char>(r * 255);
            // imageData[idx + 1] = static_cast<unsigned char>(g * 255);
            // imageData[idx + 2] = static_cast<unsigned char>(b * 255);
            // imageData[idx + 3] = 255;  // Alpha

            
            value = (value - minVal) / (maxVal - minVal + 1e-8f);  // Normalize locally

            vec3 rgb = turboColorMap(value);
            int idx = (y * kernelSize + x) * 4;

            imageData[idx + 0] = static_cast<unsigned char>(rgb.x * 255);
            imageData[idx + 1] = static_cast<unsigned char>(rgb.y * 255);
            imageData[idx + 2] = static_cast<unsigned char>(rgb.z * 255);
            imageData[idx + 3] = 255;

        }
    }

    lenia = new Lenia(width, height, totalParticles, particleRadius, spacing, {16,10} );
    leniaSize = lenia->particles.size();
    


    blockSize = 256;
    gridSize = (leniaSize + blockSize - 1) / blockSize;
    
    checkCuda(cudaMalloc(&d_leniaParticles, leniaSize * sizeof(Particle)));
    checkCuda(cudaMemcpy(d_leniaParticles, lenia->particles.data(), leniaSize * sizeof(Particle), cudaMemcpyHostToDevice));
    
    checkCuda(cudaMalloc(&d_states, blockSize * gridSize * sizeof(curandState_t)));
    
    
    checkCuda(cudaMalloc(&d_colors, colorPallete.size() * sizeof(uchar4)));
    checkCuda(cudaMemcpy(d_colors, colorPallete.data(), colorPallete.size() * sizeof(uchar4), cudaMemcpyHostToDevice));    
    


    init_random<<<gridSize, blockSize>>>(time(0), d_states);
    // checkCuda(cudaPeekAtLastError());
    // checkCuda(cudaDeviceSynchronize());
    // energy and colors
    initializeLeniaParticle<<<gridSize, blockSize>>>(d_leniaParticles, leniaSize, d_states, d_colors, colorPallete.size());
    // checkCuda(cudaPeekAtLastError());
    // checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMalloc(&d_debugU, sizeof(float)));
    checkCuda(cudaMalloc(&d_debugGrowth, sizeof(float)));
    
    
}

cudaSurfaceObject_t CUDAHandler::MapSurfaceResourse()
{
    //* Map the resource for CUDA
    cudaArray_t array;
    
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

vec3 CUDAHandler::turboColorMap(float x)
{
    x = fminf(fmaxf(x, 0.0f), 1.0f);  // Clamp to [0,1]

    const float r = 34.61f + x * (1172.33f + x * (-10793.56f + x * (33300.12f + x * (-38394.49f + x * 14825.05f))));
    const float g = 23.31f + x * (557.33f + x * (1572.88f + x * (-7743.36f + x * (10332.56f + x * -3584.14f))));
    const float b = 27.2f  + x * (3211.1f + x * (-15327.97f + x * (27814.0f + x * (-22569.18f + x * 6838.66f))));

    return vec3(r, g, b) / 255.0f;
}