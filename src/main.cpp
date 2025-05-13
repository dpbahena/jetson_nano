#include "CUDAHandler.h"

#include "GLManager.h"
#include "GL/freeglut.h"


#include <chrono>
#include <thread>

GLManager* glManager;
CUDAHandler* cudaHandler;



const int width =  1920;
const int height = 1080;

const int TARGET_FPS = 90;
const int FRAME_TIME_MS = 1000 / TARGET_FPS; // 16.67 ms per frame





void idle() {
    glutPostRedisplay();
}

void display() {
    static auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
    lastFrameTime = now;
    static int frameCount = 0;
    static float timeAccumulator = 0.0f;

    frameCount++;
    timeAccumulator += deltaTime;

    if (timeAccumulator >= 1.0f) {
        printf("Actual FPS: %d\n", frameCount);
        frameCount = 0;
        timeAccumulator = 0.0f;
    }

   



    
    // --- 2. Update simulation ---
    if (cudaHandler) {
        cudaHandler->updateDraw(deltaTime);
    }
    

    // --- 3. Render simulation scene ---
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (glManager) {
        glManager->render();
    }

    // --- 5. Render ImGui ---
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   
    // if (moveWindow && currentMonitor < monitors.size()) {
    //     MonitorInfo& m = monitors[currentMonitor];
    //     int posX = m.x + (m.width - width) / 2;  // Centered horizontally
    //     int posY = m.y + (m.height - height) / 2; // Centered vertically
    //     glutPositionWindow(posX, posY);
    //     moveWindow = false;
    // }
    

    // --- 6. FPS control ---
    auto endTime = std::chrono::high_resolution_clock::now();
    int elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - now).count() / 1000;
    int sleepTime = FRAME_TIME_MS - elapsedTime;
    if (sleepTime > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime * 1000));
    }

    glutSwapBuffers();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA OpenGL Sim");
    
    glManager = new GLManager(width, height);
    glManager->initializeGL();

    cudaHandler = new CUDAHandler(width, height, glManager->getTextureID());
    

   

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glutMainLoop();

    // Cleanup
  

 // Setup ImGui
   
}
  
