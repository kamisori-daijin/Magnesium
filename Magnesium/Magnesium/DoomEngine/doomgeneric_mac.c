//
//  doomgeneric_mac.c
//  Magnesium
//
//  Created by kamisori-daijin on 2026/08/10.
//

#include "doomgeneric.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

// Key Flag
int g_IsPressingUp = 0;
int g_IsPressingDown = 0;
int g_IsPressingLeft = 0;
int g_IsPressingRight = 0;
int g_IsPressingCtrl = 0;
int g_IsPressingSpace = 0;
int g_IsPressingEnter = 0;

uint32_t* gp_DoomScreenBuffer = NULL;

void DG_Init(void) {
    gp_DoomScreenBuffer = DG_ScreenBuffer;
}

void DG_DrawFrame(void) {}

void DG_SleepMs(uint32_t ms) {
    usleep(ms * 1000);
}

uint32_t DG_GetTicksMs(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
}

// Get Key State
int DG_GetKey(int* pressed, uint8_t* doomKey) {
    static int currentKeyIndex = 0;
    
    // Keycode
    const struct {
        int* flag;
        uint8_t code;
    } keys[] = {
        { &g_IsPressingUp, 0xad },          // Up
        { &g_IsPressingDown, 0xaf },        // Down
        { &g_IsPressingLeft, 0xac },        // Left
        { &g_IsPressingRight, 0xae },       // Right
        { &g_IsPressingCtrl, 0x80 + 0x1d },  // Ctrl
        { &g_IsPressingSpace, 0x80 + 0x39 }, // Space
        { &g_IsPressingEnter, 13 }          // Enter
    };
    
    const int numKeys = sizeof(keys) / sizeof(keys[0]);
    static int lastState[7] = {0};
    
    for (int i = 0; i < numKeys; i++) {
        int index = (currentKeyIndex + i) % numKeys;
        int currentState = *(keys[index].flag);
        
        if (currentState != lastState[index]) {
            *pressed = currentState;
            *doomKey = keys[index].code;
            lastState[index] = currentState;
            
            currentKeyIndex = (index + 1) % numKeys;
            return 1; // Ivent
        }
    }
    
    return 0; // No Ivent
}

void DG_SetWindowTitle(const char *title) {}

void mac_Doom_Create(int argc, char** argv) {
    doomgeneric_Create(argc, argv);
}

void mac_Doom_Tick(void) {
    doomgeneric_Tick();
}
