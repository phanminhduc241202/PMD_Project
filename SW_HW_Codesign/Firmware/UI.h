#pragma once

#include <Arduino.h>
#include <LiquidCrystal_I2C.h>

void initLCD();
void showMainMenu();
void showRegistrationScreen();
void showScanScreen();
void showIDScreen();
void showTimeoutMessage();
void showInvalidIDMessage();
void showInvalidCardMessage();
void showSuccessfulMessage();
void printLCD(const String& message);
