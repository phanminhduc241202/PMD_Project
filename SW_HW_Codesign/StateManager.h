#pragma once

enum class AppState {
  MAIN_MENU,
  REG_MODE,
  REG_WAIT_CARD,
  REG_WAIT_ID,
  SCAN_MODE,
  SCAN_WAIT_CARD,
  ID_MODE,
  ID_INPUT,
  SUCCESS,
  FAILURE
};

void setAppState(AppState newState);
AppState getAppState();
void handleState();