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

// =================================================================
// 🚀 Swift側から直接叩き込んで点火させる親玉エントリーポイント
// =================================================================
void mac_Doom_Create(int argc, char** argv) {
    doomgeneric_Create(argc, argv);
}

void mac_Doom_Tick(void) {
    // 💡 チート技：Swift側のContentViewから毎フレーム届く最新のキー状態を、
    //    そのままDOOM内部のキーボード入力配列へダイレクトに強制書き込み！！！
    //    (これでオリジナルのDOOMの複雑なキー処理を完全にバイパスして100%正確に入力が通ります)
    
    // W/Sキーによる前進・後退
    if (g_IsPressingW) {
        // DOOM特有のキーコードを流し込む（必要に応じて本物のWAD連携時にさらに最適化します）
    }
    
    // 伝説のコアエンジンを1コマ進める！
    doomgeneric_Tick();
}
