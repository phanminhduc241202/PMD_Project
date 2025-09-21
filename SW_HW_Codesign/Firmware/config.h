#pragma once

#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Keypad.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <AT24Cxx.h>

// RFID
#define SS_PIN 5
#define RST_PIN 0

// Alerts
#define RED_LED 16
#define GREEN_LED 4
#define BUZZER 17

// Keypad
#define R1 26
#define R2 27
#define R3 14
#define R4 12
#define C1 13
#define C2 32
#define C3 33
#define C4 25

const byte ROWS = 4;
const byte COLS = 4;
char hexaKeys[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};
byte rowPins[ROWS] = {R1, R2, R3, R4};
byte colPins[COLS] = {C1, C2, C3, C4};

// Hardware Objects
extern MFRC522 mfrc522;
extern Keypad customKeypad;
extern LiquidCrystal_I2C lcd;
extern AT24Cxx mem;

void initHardware();
