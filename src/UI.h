#pragma once

#include <fstream>
#include <vector>


class CUDAHandler;  // Just temporary declaration but necessary

struct Snapshot {
    std::string description;
    int totalParticles;
    float convolutionRadius;
    float alpha;
    float sigma;
    float mu;
    float m;
    float s;
    float conv_dt;
    int gMode;
    int kMode;
};


class SimulationUI {
    public:

        SimulationUI();
        ~SimulationUI();
        void render(CUDAHandler &sim); // * real function
        void handleKeypress(unsigned char key, int x, int y); //* real function
        void handleMouseClick(int button, int state, int x, int y);
        static void keyboardCallback(unsigned char key, int x, int y); //* static wrapper
        static void mouseCallback(int button, int state, int x, int y); //* wrapper

        static void mouseMotionCallback(int x, int y);
        static void mouseWheelCallback(int wheel, int direction, int x, int y);

        

        static SimulationUI* instance;  //* Singleton-like access

        // save and load functions
        std::ofstream paramLogFile;
        char descriptionBuffer[256] = "";  // buffer for text input
        std::vector<Snapshot> savedSnapshots;
        std::string selectFilename = "";
        int selectedSnapshotIndex = -1;
        void loadSnapshotsFromFile(const std::string& filename);
        
        bool showMenu = true;
        float cellSize = 8.f;


};