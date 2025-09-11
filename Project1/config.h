#ifndef CONFIG_H
#define CONFIG_H

// Khai báo địa chỉ LCD I2C
#define LCD_ADDRESS 0x27

// Khai báo chân module thu hồng ngoại
#define PIN_RECEIVER 34

// Sử dụng mảng để quản lý các chân Relay và nút nhấn
const int RELAY_PINS[] = {16, 17, 5, 18};
const int BUTTON_PINS[] = {25, 33, 32, 35};
const int NUM_DEVICES = 4;

#endif