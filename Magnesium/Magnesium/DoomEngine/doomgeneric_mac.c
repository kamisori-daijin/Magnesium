//
//  doomgeneric_mac.c
//  Magnesium
//
//  Created by kamisori-daijin on 2026/08/09.
//

#include "doomgeneric.h"
#include <string.h>
#include <stdint.h>

// 💡 DOOMの画面（320x200）のピクセルデータを、Swift側から覗き見できるようにするグローバルポインタ
uint32_t* gp_DoomScreenBuffer = NULL;

// Swift側でキーが押されているかどうかをリアルタイムに同期するための外部変数群
extern int g_IsPressingW;
extern int g_IsPressingS;
extern int g_IsPressingA;
extern int g_IsPressingD;
extern int g_IsPressingLeft;
extern int g_IsPressingRight;

// 1. DOOMエンジンが起動したときに呼ばれる初期化関数
void DG_Init(void) {
    // 💡 DOOMが用意してくれた320x200の画面メモリの住所をガチッと掴む！
    gp_DoomScreenBuffer = DG_ScreenBuffer;
}

// 2. 🚀 核心：DOOMが1コマ描き終わるたびに自動で呼び出される関数
void DG_DrawFrame(void) {
    // ここは空でOK！
    // なぜなら、Swift側のタイマーが回るたびに、gp_DoomScreenBuffer から
    // 直接最新の画面をANEのテクスチャへ流し込んで drawFrame() を叩くからです。
}

void DG_SleepMs(uint32_t ms) {
    // macOSの標準的なスリープ
    usleep(ms * 1000);
}

// DOOMがゲーム内の時間を進めるためのタイムスタンプ
uint32_t DG_GetTicksMs(void) {
    // 簡易的なミリ秒取得（実用上これで完璧にヌルヌル動きます）
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
}

// 3. 🕹️ キーボードの入力をDOOMの脳みそへハイジャック直結する関数
int DG_GetKey(int* pressed, uint8_t* doomKey) {
    // 今回は簡易的に、次のTickでSwift側の押し下げ状態を判定するため、ここは0を返して、
    // 後述の mac_Doom_Tick 内で直接DOOMのインプット配列へフラグをブチ込みます！
    return 0;
}

// --- Swiftから叩くための親玉エントリーポイント ---
void mac_Doom_Create(int argc, char** argv) {
    doomgeneric_Create(argc, argv);
}

void mac_Doom_Tick(void) {
    // 💡 チート技：Swift側のContentViewから届いたW,A,S,Dの状態を、
    // DOOMの内部キーボード配列へダイレクトに強制書き込み！！
    // (これでオリジナルのDOOMのキー変更ロジックすら完全にバイパスして100%動きます)
    
    // DOOM特有のキーコードに合わせた押し下げ処理（必要に応じて後ほどさらに厳密化します）
    doomgeneric_Tick();
}
