#include "config.h"
#include "UI.h"
#include "StateManager.h"

void setup() {
  initHardware();
  initLCD();
  setAppState(AppState::MAIN_MENU);
}

void loop() {
  handleState();
}