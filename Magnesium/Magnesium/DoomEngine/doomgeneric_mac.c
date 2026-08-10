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


int g_IsPressingW = 0;
int g_IsPressingS = 0;
int g_IsPressingA = 0;
int g_IsPressingD = 0;
int g_IsPressingLeft = 0;
int g_IsPressingRight = 0;
int g_IsPressingEnter = 0;
int g_IsPressingSpace = 0;

// DOOMの画面（320x200）のピクセルデータを、Swift側（ANERenderer）から覗き見できるようにするグローバルポインタ
uint32_t* gp_DoomScreenBuffer = NULL;

// 1. DOOMエンジンが起動したときに1回だけ呼ばれる初期化関数
void DG_Init(void) {
    // DOOMが用意してくれた320x200の画面メモリの住所をガチッと掴む！
    gp_DoomScreenBuffer = DG_ScreenBuffer;
}

// 2. DOOMが1コマ描き終わるたびに自動で呼び出される関数（空で100%OK）
void DG_DrawFrame(void) {
}

// OSのスリープ（ミリ秒）
void DG_SleepMs(uint32_t ms) {
    usleep(ms * 1000);
}

// DOOMがゲーム内の時間を正確に進めるためのタイムスタンプ関数
uint32_t DG_GetTicksMs(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
}

// 3. キーボードのインプット窓口
int DG_GetKey(int* pressed, uint8_t* doomKey) {
    return 0;
}

// 4. ウィンドウタイトル設定（スタブとして空実装）
void DG_SetWindowTitle(const char *title) {
}



void mac_Doom_Create(int argc, char** argv) {
    doomgeneric_Create(argc, argv);
}

extern void D_PostEvent(void* ev);

void mac_Doom_Tick(void) {
   
    typedef struct {
        int type;     // 1 = ev_keydown, 2 = ev_keyup
        int data1;    // KeyCode
        int data2;
        int data3;
    } doom_key_event_t;

    
    static int lastEnterState = 0;
    static int autoEnterCount = 0;
    if (autoEnterCount < 1000) {
        doom_key_event_t ev = {1, 13, 0, 0}; // KEY_ENTER 
        D_PostEvent(&ev);
        autoEnterCount++;
    }

    //  W / S
    static int lastW = 0;
    if (g_IsPressingW != lastW) {
        // 119 = 'w' (KEY_STRAFE_FORWARD)
        doom_key_event_t ev = { g_IsPressingW ? 1 : 2, 119, 0, 0 };
        D_PostEvent(&ev);
        lastW = g_IsPressingW;
    }
    
    static int lastS = 0;
    if (g_IsPressingS != lastS) {
        // 115 = 's' (KEY_STRAFE_BACKWARD)
        doom_key_event_t ev = { g_IsPressingS ? 1 : 2, 115, 0, 0 };
        D_PostEvent(&ev);
        lastS = g_IsPressingS;
    }


    static int lastLeft = 0;
    if (g_IsPressingLeft != lastLeft) {
        // 0xac = KEY_LEFTARROW 
        doom_key_event_t ev = { g_IsPressingLeft ? 1 : 2, 0xac, 0, 0 };
        D_PostEvent(&ev);
        lastLeft = g_IsPressingLeft;
    }

    static int lastRight = 0;
    if (g_IsPressingRight != lastRight) {
        // 0xae = KEY_RIGHTARROW
        doom_key_event_t ev = { g_IsPressingRight ? 1 : 2, 0xae, 0, 0 };
        D_PostEvent(&ev);
        lastRight = g_IsPressingRight;
    }


    doomgeneric_Tick();
}
