#pragma once

#include <vector>
#include <string>

struct MonitorInfo {
    int x, y;
    int width, height;
    std::string name;
};

extern std::vector<MonitorInfo> monitors;
extern int currentMonitor;
extern bool moveWindow;
extern int width, height; // make sure these are defined in your main .cpp
void findMonitors();
