#include "UI.h"

#include "CUDAHandler.h"
#include "GLManager.h"


SimulationUI* SimulationUI::instance = nullptr;


void SimulationUI::render(CUDAHandler &sim)
{
    // Toggle with a TAB key to show menu
    // if (ImGui::IsKeyPressed(ImGuiKey_Tab)) {
    //     showMenu = ! showMenu;
    // }


    if (showMenu) {

        ImGui::Begin("Simulation Control");
        if (ImGui::Button("Reset Sim")) {
            #if defined(__aarch64__) || defined(USE_X11_MONITORS)
            sim.totalParticles = 15e4;
            #else
            sim.totalParticles = 1e6;
            #endif
            
            sim.spacing = 1.0f;
            sim.particleRadius = .5f;
            sim.initLenia();
        }
        if (sim.startSimulation)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.8f, 0.2f, 1.0f)); // green
        else
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f)); // red

        ImGui::Text("Start Simulation :"); ImGui::SameLine();
        if (ImGui::Button(sim.startSimulation ? "ON" : "OFF")) {
            sim.startSimulation ^= true;
        }
        ImGui::PopStyleColor();


        // // Combo for Tree Display Mode
        // const char* options[] = { "Grid", "Vertical", "Horizontal", "Checkered", "Diagonal", "X-shape", "Circle", "Spiral", "Border", "Doble border", "Rings", "Radial", "Animated Beams", "Diagonals Grid", "Full Grid", "Cellular Automata"};
        // static int selectedOption = sim.option; 

        // if (ImGui::Combo("Game of Life Pattern", &selectedOption, options, IM_ARRAYSIZE(options))) {
        //     sim.option = selectedOption;
        // }
        #if defined(__aarch64__) || defined(USE_X11_MONITORS)
        ImGui::SliderInt("Number of Particles", &sim.totalParticles, 10000, 200000);
        #else
        ImGui::SliderInt("Number of Particles", &sim.totalParticles, 10000, 2000000);
        #endif
        ImGui::PushItemWidth(150);
        ImGui::SliderFloat("Radius", &sim.particleRadius, 0.1f, 30.f);
        ImGui::SliderFloat("distance", &sim.spacing, .2f, 30.f);        
        ImGui::SliderInt("R", &sim.convolutionRadius, 1, 12);
        if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Convolution Radius");
        ImGui::SliderFloat("Alpha", &sim.alpha, 2.0f, 6.0f);
        ImGui::Separator();
        float step = .001f;
        float step2 = .0001f;
        
        ImGui::SliderFloat("S", &sim.sigma, .01f, 0.08f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sigma");
        
        // Fine-tune buttons for sigma
        if (ImGui::Button("-##sigma1")) {
            sim.sigma -= step;
        }
        ImGui::SameLine();
        if (ImGui::Button("+##sigma1")) {
            sim.sigma += step;
        }
        ImGui::SameLine();
        if (ImGui::Button("-##sigma2")) {
            sim.sigma -= step2;
        }
        ImGui::SameLine();
        if (ImGui::Button("+##sigma2")) {
            sim.sigma += step2;
        }
        
        float muStep = .001f;
        float muStep2 = .0001f;
        
        ImGui::SliderFloat("mu", &sim.mu, .014f, 0.28f);
        
        // Fine-tune buttons for mu
        if (ImGui::Button("-##mu1")) {
            sim.mu -= muStep;
        }
        ImGui::SameLine();
        if (ImGui::Button("+##mu1")) {
            sim.mu += muStep;
        }
        ImGui::SameLine();
        if (ImGui::Button("-##mu2")) {
            sim.mu -= muStep2;
        }
        ImGui::SameLine();
        if (ImGui::Button("+##mu2")) {
            sim.mu += muStep2;
        }
        
        ImGui::SliderFloat("Pick of Ring", &sim.m, .01f, 0.9f);
        ImGui::SliderFloat("Thickness/Ring Spread", &sim.s, .01f, 0.21f);
        ImGui::SliderFloat("DT", &sim.conv_dt, 0.001, 0.15);
        ImGui::PopItemWidth();


        // int gameMode = static_cast<GameMode>(sim.gameMode);
        // ImGui::RadioButton("Game Of Life", &gameMode, gameOfLife); ImGui::SameLine();
        // ImGui::RadioButton("TanH", &gameMode, hyperbolicTanF); ImGui::SameLine();
        // ImGui::RadioButton("Sigmoid", &gameMode, sigmoidF); ImGui::SameLine();
        // ImGui::RadioButton("reLu", &gameMode, reLuF); ImGui::SameLine();
        // sim.gameMode = (GameMode)gameMode;
        // ImGui::NewLine();
        // if (gameMode == sigmoidF || gameMode == hyperbolicTanF || gameMode == reLuF) {
            
        //     ImGui::SliderFloat("Threshold", &sim.sigmoidThreshold, 0.0f, 8.0f);
        //     if (ImGui::CollapsingHeader("Convolution Kernel2")) {
        //         ImGui::Text("Editable 3x3 Kernel");
            
        //         for (int row = 0; row < 3; ++row) {
        //             ImGui::PushID(row);  // isolate slider IDs
        //             for (int col = 0; col < 3; ++col) {
        //                 ImGui::PushID(col);
        //                 ImGui::SetNextItemWidth(50);
        //                 ImGui::SliderFloat("##", &sim.kernelMatrix[row * 3 + col], -2.0f, 2.0f);
        //                 ImGui::PopID();
        //                 ImGui::SameLine();
        //             }
        //             ImGui::PopID();
        //             ImGui::NewLine();
        //         }
        //     }
        // }
        // ImGui::SliderFloat("KernelSigma", &sim.kernelSigma, .01f, 15.15f);
        // ImGui::SliderFloat("radius kernel", &sim.kernelRadius, 1.0f, 15.0f);
        // ImGui::SliderFloat("sigma", &sim.sigma, .001f, 0.05f);
        // ImGui::SliderFloat("mu", &sim.mu, .0111f, 0.22f);
        // ImGui::SliderFloat("DT", &sim.conv_dt, 0.009, 0.20);
        // if (sim.option == 0) { // grid
        //     ImGui::SliderInt("Grid Size", &sim.gridSize, 2, 100);
        // } 
        // if (sim.option > 0 && sim.option < 4) {
        //     ImGui::SliderFloat("Size", &sim.widthFactor, 0.01f,1.0f);
        // }
        // if (sim.option == 10 ) {
        //     ImGui::SliderFloat("Spacing", &sim.ringSpacing, 5.0f,100.0f);
        //     ImGui::SliderFloat("Thickness", &sim.thickness, 1.0f,20.0f);
        // }
        // if (sim.option == 7) {
        //     ImGui::SliderFloat("Spacing", &sim.spacing, 5.0f,70.0f);
        //     ImGui::SliderFloat("Thickness", &sim.thickness, 1.0f,20.0f);
        // }
        // if (sim.option == 13) {
        //     ImGui::SliderInt("BlockSize", &sim.blockSize, 5.0f,70.0f);
        //     ImGui::SliderInt("Band", &sim.band, 1.0f,20.0f);
        // }
        // if (sim.option == 14) {
        //     ImGui::SliderInt("BlockSize", &sim.blockSize, 5.0f,70.0f);
        //     ImGui::SliderInt("DiagonalBand", &sim.diagonalBand, 1.0f,20.0f);
        //     ImGui::SliderInt("Border", &sim.border, 1.0f,20.0f);
        // }
        // if (sim.option ==15) {
        //     ImGui::PushID("Rule Cell Automata");
        //     static int ruleSlider = 30;
        //     ImGui::SliderInt("##Slider", &ruleSlider, 0, 255, "Rule: %0d");
        //     ImGui::SameLine();
        //     ImGui::InputInt("##Input", &ruleSlider);
        //     sim.rule = static_cast<uint8_t>(ruleSlider);
        //     ImGui::PopID();
        // }

        static float shellData[100];
        for (int i = 0; i < 100; ++i) {
            float r = i / 99.0f;
            float val = 0.0f;
            if (r > 0.0f && r < 1.0f)
                val = expf(sim.alpha - sim.alpha / (4.0f * r * (1.0f - r)));
            shellData[i] = val;
        }
        ImGui::PlotLines("Shell Kernel Slice", shellData, 100, 0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 100));

        // Plot the kernel: K(r) = exp(-(r - m)^2 / (2 * s^2)), r in [0, 1]
        static float kernelData[100];
        for (int i = 0; i < 100; ++i) {
            float r = i / 99.0f; // r in [0,1]
            kernelData[i] = expf(-((r - sim.m) * (r - sim.m)) / (2.0f * sim.s * sim.s));
        }
        ImGui::PlotLines("Kernel Slice K(r)", kernelData, 100, 0, nullptr, 0.0f, 1.0f, ImVec2(0, 100));

        // draw actual kernel
        const std::vector<float>& kernel = sim.convKernel;  // Already generated and normalized
        int kernelSize = sim.kernelSize;
        if (ImGui::CollapsingHeader("Kernel 2D View")) {
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 canvas_pos = ImGui::GetCursorScreenPos(); // Top-left of the canvas
            float cellSize = 8.0f;  // Adjust size to taste

            for (int y = 0; y < kernelSize; ++y) {
                for (int x = 0; x < kernelSize; ++x) {
                    float value = kernel[y * kernelSize + x]; // Assuming row-major order
                
                    // Clamp and map to 0-255 grayscale
                    float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
                    int shade = static_cast<int>(clamped * 255.0f);
                
                    ImU32 color = IM_COL32(shade, shade, shade, 255);
                    ImVec2 p0 = ImVec2(canvas_pos.x + x * cellSize, canvas_pos.y + y * cellSize);
                    ImVec2 p1 = ImVec2(p0.x + cellSize, p0.y + cellSize);
                    draw_list->AddRectFilled(p0, p1, color);
                    draw_list->AddRect(p0, p1, IM_COL32(50, 50, 50, 255)); // thin grid line border

                }
            }
        
            // Reserve space in the layout
            ImGui::Dummy(ImVec2(cellSize * kernelSize, cellSize * kernelSize));
        }

        // gradient color instead
    
        // GLuint kernelTextureID;
        // glGenTextures(1, &kernelTextureID);
        // glBindTexture(GL_TEXTURE_2D, kernelTextureID);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sim.kernelSize, sim.kernelSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, sim.imageData.data());
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Once kernelData is ready:
        GLManager glManager(sim.kernelSize, sim.kernelSize);
        glManager.uploadTexture(sim.imageData.data(), sim.kernelSize, sim.kernelSize);
        GLuint kernelTextureID = glManager.getTextureID();
        ImGui::Text("Kernel Heatmap Texture");
        // ImGui::Image((ImTextureID)(intptr_t)kernelTextureID, ImVec2(200, 200));  // Adjust size as needed

        
        ImGui::SliderFloat("CellSize", &cellSize, 3.f, 15.f);
        float viewSize = sim.kernelSize * cellSize;
        ImGui::Image((ImTextureID)(intptr_t)kernelTextureID, ImVec2(viewSize, viewSize));
        





        ImGui::Text("Latest u: %.4f", sim.debugU_host);
        ImGui::Text("Latest growth: %.4f", sim.debugGrowth_host);
        // printf("size %d\n",(int)sim.uHistory.size());
        if (!sim.uHistory.empty())
            ImGui::PlotLines("Excitation (u)", sim.uHistory.data(), sim.uHistory.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(0, 80));
        if (!sim.growthHistory.empty())
            ImGui::PlotLines("Growth", sim.growthHistory.data(), sim.growthHistory.size(), 0, nullptr, -1.0f, 1.0f, ImVec2(0, 80));

        ImGui::Separator;
        ImGui::Text("Total Cells: %d", (int)sim.leniaSize);
        ImGui::Text("Zoom Factor: %f", sim.zoom);
        ImGui::SliderInt("FPS", &sim.TARGET_FPS, 20, 120);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Actual FPS: %.1f", 1.0f / sim.dt);
        
        
        // Add more sliders for simulation params here
        ImGui::End();

    }
}  
    
    


    // if (showMenu) {

    //     ImGui::Begin("Simulation Control");
        
        // const char* modes[] = {"Editing", "Simulating"};
        // static int currentMode = 0;
        // if (ImGui::Combo("Mode", &currentMode, modes, IM_ARRAYSIZE(modes))) {
        //     sim.setMode(currentMode == 0 ? SimMode::Editing : SimMode::Simulating);
        // }
        // ImGui::Separator;

        // if (sim.isEditing()) {
        //     ImGui::Text("Editor Mode: add planets, set orbits.");
        //     // TODO: future add buttons to place/edit bodies
        // } else {
        //     ImGui::Text("Simulation MOde: gravity in action.");
        //     // TODO: add sliders or pause control
        // }
        // if (ImGui::Button("Reset Sim")) {
        //     sim.resetSimulation();
        // }
        // if (ImGui::IsItemHovered()) {
        //     ImGui::SetTooltip("Remove all particles & reset the simulation");
        // }
        // if (sim.getTreeMode() == NBODY) {
        //     ImGui::SliderInt("Trail Length", &sim.trailLength, 1, 400);
        // }else {
        //     ImGui::SliderInt("Trail Length", &sim.trailLength, 1, 60);
        // }

        // ImGui::SliderInt("Stiffness", &sim.stiffness, 1, 1500);
        // ImGui::SliderFloat("StringLength", &sim.restLength, 0.1, 200.0f);
        // ImGui::SliderFloat("Radius", &sim.particleRadius, 0.5f, 20.0f);
        // ImGui::SliderFloat("Mass", &sim.particleMass, 0.1f, 10.0f);
        // ImGui::SliderFloat("Damping", &sim.damping, 1.0f, 300.0f);
        // ImGui::SliderFloat("Kpressure", &sim.kp, 0.01f, 10.0f);
        // ImGui::SliderFloat("Correction", &sim.posCorrection, 0.01f, 0.5f);
        // ImGui::SliderInt("# of Correction Passes", &sim.posCorrectionPasses, 0, 10);
        // ImGui::SliderInt("Overlap correction passes", &sim.overlapCorrectionPasses, 0, 20);
        // if(sim.getTreeMode() == NBODY) {
        //     ImGui::SliderFloat("G Value", &sim.gValue, 1, 5000);
        //     ImGui::SliderFloat("Min Value", &sim.minValue, 0, 1000);
        //     ImGui::SliderFloat("Max VAlue", &sim.maxValue, 1, 5000);
        // }
        // if(sim.getTreeMode() == SPHERE) {
        //     ImGui::SliderInt("Circular Boundary", &sim.circularBoundary, 100, 500);
        //     ImGui::SliderFloat("Restitution", &sim.restitution, -0.5f, 2.0f);
        //     ImGui::SliderFloat("Friction", &sim.friction, -0.5f, 2.0f);
        // }
        // ImGui::PushID("Drag");
        // ImGui::SliderFloat("##Slider", &sim.dragCoeff, 0.000f, 2.0f, "Drag Coef: %.3f");
        // ImGui::SameLine();
        // ImGui::InputFloat("##Input", &sim.dragCoeff);
        // ImGui::PopID();
        // ImGui::SliderFloat("West wind", &sim.wind, -20.0f, 20.0f);
        // ImGui::SliderFloat("Valley breeze", &sim.valleyBreeze, 0.f, 40.0f);
        // ImGui::SliderInt("Cursor Radius", &sim.mouseCursorRadius, 5, 50);
        // ImGui::Checkbox("Gravity", sim.getGravityPtr());
        // ImGui::Checkbox("FollowMode", sim.getFollowModePtr());
        
        // int toolMode = static_cast<ToolMode>(sim.getToolMode());
        // ImGui::Text("Select tool:"); 
        // ImGui::RadioButton("Disturbe", &toolMode, DISTURBE); ImGui::SameLine();
        // ImGui::RadioButton("Tear", &toolMode, TEAR); ImGui::SameLine();
        // ImGui::RadioButton("Pull/drag", &toolMode, DRAG); ImGui::SameLine();
        // ImGui::RadioButton("Activate", &toolMode, ACTIVE);

        
        
        // sim.setToolMode(toolMode);
        
        // // Combo for Tree Display Mode
        // const char* treeModes[] = { "Cycle Tree", "Complete Tree", "In Line", "Grid", "Single", "Nbody", "Spider", "Pendulum", "Sphere"};
        // static int selectedTreeMode = sim.getTreeMode(); // 0: Cycle, 1: Complete, 2: In Line

        // if (ImGui::Combo("Tree Mode", &selectedTreeMode, treeModes, IM_ARRAYSIZE(treeModes))) {
        //     sim.setTreeMode(selectedTreeMode);
        // }
        // ImGui::Checkbox("Show Links", sim.getShowLinksPtr());
        // ImGui::Checkbox("Show Particles", sim.getShowParticlesPtr());
        // if (sim.getTreeMode() == GRID) {
        //     ImGui::Checkbox("Show Texture", sim.getTextureModPtr());
        // }
        // ImGui::Checkbox("Windy", &sim.windy);


        // ImGui::Checkbox("Show Trails", sim.getShowTrailsPtr());
        // ImGui::Text("# of Particles: %d",(int)sim.particles.size());
        // ImGui::Text("Zoom Factor: %f", sim.zoom);
    //     ImGui::Separator;
    //     ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    //     ImGui::Text("Actual FPS: %.1f", 1.0f / sim.dt);
        

    //     // Add more sliders for simulation params here
    //     ImGui::End();
    // }
// }

// void SimulationUI::handleKeypress(unsigned char key, int x, int y)
// {
//     ImGuiIO& io = ImGui::GetIO();
//     if (io.WantCaptureKeyboard) return;

//     if (key == 27){
//         exit(0); 
//     }
//     // ? Handle other keys
// }

// void SimulationUI::keyboardCallback(unsigned char key, int x, int y)
// {
//     // Step 1: let ImGui handle the keyboard input
//     ImGui_ImplGLUT_KeyboardFunc(key, x, y);

//     // Step 2: Pass it to your own logic
//     if (instance){
//         instance->handleKeypress(key, x, y);
//     }
// }

// void SimulationUI::mouseCallback(int button, int state, int x, int y)
// {
//     // Step 1: Let ImGui handle input (so UI stays interactive)
//     ImGui_ImplGLUT_MouseFunc(button, state, x, y);

//     // Step 2: Pass it to your own logic
//     if (instance){
//         instance->handleMouseClick(button, state, x, y);
//     }


// }

// void SimulationUI::mouseMotionCallback(int x, int y)
// {
    
//     CUDAHandler* sim = CUDAHandler::instance;
//     // * Let ImGui handles mouse events first
//     ImGui_ImplGLUT_MotionFunc(x, y);

//     ImGuiIO& io = ImGui::GetIO();
//     if (io.WantCaptureMouse || !sim) return;

//     int mods = glutGetModifiers();
//     if (sim->isDragging  && sim->isPanEnabled) {
//         float dx = (x - sim->lastMouseX) / sim->zoom;
//         float dy = (y - sim->lastMouseY) / sim->zoom;

//         sim->panX += dx;
//         sim->panY += dy;

//         sim->lastMouseX = x;
//         sim->lastMouseY = y;

//         glutPostRedisplay();
//     } else if (sim->toolMode == DISTURBE) {
//         // Convert screen coordinates to world space
//         float worldx = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
//         float worldy = (y - sim->height / 2.0f) / sim->zoom - sim->panY;
//         vec2 targetPos(worldx, worldy);
//         sim->disturbeGameLife(targetPos);
//         // record screen position
//         // sim->lastMouseX = x;
//         // sim->lastMouseY = y;

//     }
//     sim->lastMouseX = x;
//     sim->lastMouseY = y;

// }

// void SimulationUI::mouseWheelCallback(int wheel, int direction, int x, int y)
// {
//     if (!CUDAHandler::instance) return;
//     int mods = glutGetModifiers();  // activate zoom/via wheel mouse if ALT key is pressed
//     if (mods & GLUT_ACTIVE_SHIFT) { 
//         float zoomFactor = (direction > 0) ? 1.1f : 1.0f / 1.1f;
//         CUDAHandler::instance->zoom *= zoomFactor;
//         glutPostRedisplay();
//     }
// }

// void SimulationUI::handleMouseClick(int button, int state, int x, int y){
    
//     ImGuiIO& io = ImGui::GetIO();
//     if (io.WantCaptureMouse || !CUDAHandler::instance) return;

//     CUDAHandler* sim = CUDAHandler::instance;
    

//     // ✅ Convert screen coordinates to world/sim space
//     float worldX = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
//     float worldY = (y - sim->height / 2.0f) / sim->zoom - sim->panY;
//     // vec2f mousePos(worldX, worldY);
//     if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {

//         int mods = glutGetModifiers();
//         if (mods & GLUT_ACTIVE_SHIFT) { // onnly activate dragging if SHIFT key is pressed
//             sim->isDragging = true;
//             sim->isPanEnabled = true;
//             sim->lastMouseX = x;
//             sim->lastMouseY = y;
            

//         } else {
//             sim->toolMode = DISTURBE;
//         }

//         // if (sim->getToolMode() == DRAG)
//         //     sim->applyPullToParticle(mousePos); // grab a particle
        
       
//         // sim->lastMouseX = x; // This is used to update the circular mouse pointer 
//         // sim->lastMouseY = y; 

//         // sim->leftMouseDown = true;S

        
//     } else if (state == GLUT_UP) {
//         sim->isDragging = false;
//         sim->isPanEnabled = false;
//         // sim->leftMouseDown = false;
//         // sim->selectedParticle = nullptr;
//         // for (int i = 0; i < sim->selectedParticles.size(); ++i){
//         //     sim->particles[sim->selectedParticles[i]].color = sim->simulationParticlesColor;
//         // }
//         // sim->selectedParticles.clear();
//         // sim->setToolMode(DISTURBE); // default

//     }

//     // if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {

//     //     // ✅ Convert screen coordinates to world/sim space
//     //     // float worldX = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
//     //     // float worldY = (y - sim->height / 2.0f) / sim->zoom - sim->panY;

//     //     // float mass = random_double(1.0f, 10.0);
//     //     // float radius = random_double(5.0, 20.0);
//     //     // float indexColor = random_int(0, sim->colorPalette.size());

//     //     // Particle ball(Vec2(worldX, worldY), mass, radius , sim->colorPalette[indexColor], 0);
//     //     // if (!sim->particles.empty()) {
//     //     //     // ball.velocity = Vec2(0.0f, 0.0f);

//     //     //     sim->particles.push_back(ball);
        
//     //     //     // update trail size
//     //     //     sim->particleTrail.resize(sim->particles.size());
//     //     //     // sim->planetAngle.resize(sim->particles.size());
//     //     // }
        
           

//     // }
// }


