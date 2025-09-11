#ifndef IR_CONTROL_H
#define IR_CONTROL_H

#include "config.h"
#include <LiquidCrystal_I2C.h>
#include <IRremote.h>
#include <virtuabotixRTC.h>

// Khai báo các đối tượng extern để các file khác có thể sử dụng
extern LiquidCrystal_I2C lcd;
extern IRrecv receiver;
extern virtuabotixRTC myRTC;

// Định nghĩa kiểu con trỏ hàm (Callback)
typedef void (*IrCommandCallback)();

// Khai báo hàm
void initIRReceiver();
void translateIR();

#endif