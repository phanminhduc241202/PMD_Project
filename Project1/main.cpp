#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "relay_control.h"
#include "button_control.h"
#include "ir_control.h"

void setup() {
    Serial.begin(115200);
    Wire.begin();
    lcd.begin();
    lcd.clear();
    
    initRelays();
    initButtons();
    initIRReceiver();
    
    // Set thời gian trên DS1302
    myRTC.setDS1302Time(00, 32, 03, 3, 18, 2, 2024);
    
    // Màn hình khởi động
    lcd.setCursor(0, 0);
    lcd.print("System Started!");
}

void loop() {
    read_button();
    
    if (receiver.decode()) {
        translateIR();
        receiver.resume();
    }
    
    delay(100);
}