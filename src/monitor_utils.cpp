#include "monitor_utils.h"

#include <cstdio>

std::vector<MonitorInfo> monitors;
int currentMonitor = 0;
bool moveWindow = false;

// Define either based on platform or in CMakeLists.txt
#if defined(__aarch64__) || defined(USE_X11_MONITORS)

#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include <GL/freeglut.h>

void findMonitors() {
    Display* dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "Cannot open X display\n");
        return;
    }

    Window root = DefaultRootWindow(dpy);
    XRRScreenResources* res = XRRGetScreenResourcesCurrent(dpy, root);
    if (!res) {
        fprintf(stderr, "Failed to get screen resources\n");
        XCloseDisplay(dpy);
        return;
    }

    for (int i = 0; i < res->noutput; ++i) {
        XRROutputInfo* outputInfo = XRRGetOutputInfo(dpy, res, res->outputs[i]);
        if (outputInfo && outputInfo->connection == RR_Connected && outputInfo->crtc) {
            XRRCrtcInfo* crtcInfo = XRRGetCrtcInfo(dpy, res, outputInfo->crtc);
            if (crtcInfo) {
                MonitorInfo m;
                m.x = crtcInfo->x;
                m.y = crtcInfo->y;
                m.width = crtcInfo->width;
                m.height = crtcInfo->height;
                m.name = outputInfo->name ? outputInfo->name : "Unknown";
                monitors.push_back(m);
                XRRFreeCrtcInfo(crtcInfo);
            }
        }
        if (outputInfo) XRRFreeOutputInfo(outputInfo);
    }

    XRRFreeScreenResources(res);
    XCloseDisplay(dpy);

    if (!monitors.empty()) {
        MonitorInfo& m = monitors[std::min(1, (int)monitors.size() - 1)];
        glutInitWindowPosition(m.x + (m.width - width) / 2, m.y + (m.height - height) / 2);
        currentMonitor = std::min(1, (int)monitors.size() - 1);
    }
}

#else // SDL fallback for desktop

#include <SDL2/SDL.h>
#include <GL/freeglut.h>

void findMonitors() {
    SDL_Init(SDL_INIT_VIDEO);
    int numDisplays = SDL_GetNumVideoDisplays();
    for (int i = 0; i < numDisplays; ++i) {
        SDL_Rect bounds;
        if (SDL_GetDisplayBounds(i, &bounds) == 0) {
            MonitorInfo m;
            m.x = bounds.x;
            m.y = bounds.y;
            m.width = bounds.w;
            m.height = bounds.h;
            m.name = "Monitor " + std::to_string(i + 1);
            monitors.push_back(m);
        }
    }

    if (!monitors.empty()) {
        MonitorInfo& m = monitors[std::min(1, (int)monitors.size() - 1)];
        glutInitWindowPosition(m.x + (m.width - width) / 2, m.y + (m.height - height) / 2);
        currentMonitor = std::min(1, (int)monitors.size() - 1);
    }

    SDL_Quit();
}

#endif
