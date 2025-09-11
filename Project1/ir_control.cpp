#include "ir_control.h"
#include "relay_control.h"

LiquidCrystal_I2C lcd(LCD_ADDRESS, 16, 2);
IRrecv receiver(PIN_RECEIVER);
virtuabotixRTC myRTC(27, 14, 12); // (CLK, DAT, RST)

// Các hàm xử lý riêng cho từng lệnh IR
//Relay 1
void handle_relay1_ir() 
{ 
    control_relay(0); 
    lcd.print("Relay 1 toggled"); 
}
//Relay 2
void handle_relay2_ir() 
{ 
    control_relay(1); 
    lcd.print("Relay 2 toggled"); 
}
//Relay 3
void handle_relay3_ir() 
{ 
    control_relay(2); 
    lcd.print("Relay 3 toggled"); 
}
//Relay 4
void handle_relay4_ir() 
{ 
    control_relay(3); 
    lcd.print("Relay 4 toggled"); 
}

void handle_display_time_ir() {
    myRTC.updateTime();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(myRTC.dayofmonth); lcd.print("/");
    lcd.print(myRTC.month);      lcd.print("/");
    lcd.print(myRTC.year);
    lcd.setCursor(0, 1);
    lcd.print(myRTC.hours);      lcd.print(":");
    lcd.print(myRTC.minutes);    lcd.print(":");
    lcd.print(myRTC.seconds);
}

void handle_all_relay_off_ir() {
    turn_off_all_relays();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("KEY# pressed");
    lcd.setCursor(0, 1);
    lcd.print("ALL Relay OFF");
}

// Cấu trúc để ánh xạ mã lệnh IR và hàm xử lý tương ứng
struct IrMapping {  // tong kich thuoc 8 byte
    long command;
    IrCommandCallback callback;
};

// Mảng các ánh xạ, sử dụng con trỏ hàm
IrMapping ir_mappings[] = {
    {8, handle_relay1_ir}, // ma nut so 1
    {9, handle_relay2_ir}, // ma nut so 2
    {10, handle_relay3_ir}, // ma nut so 3
    {11, handle_relay4_ir}, // ma nut so 4
    {22, handle_display_time_ir}, // nut *
    {13, handle_all_relay_off_ir}, // nut #
};
const int NUM_IR_COMMANDS = sizeof(ir_mappings) / sizeof(IrMapping); // Tính toán số lệnh IR trong mảng

void initIRReceiver() {
    receiver.enableIRIn();
}

void translateIR() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("button pressed:");
    lcd.setCursor(0, 1);

    long command = receiver.decodedIRData.command;
    for (int i = 0; i < NUM_IR_COMMANDS; i++) {
        if (ir_mappings[i].command == command) {
            ir_mappings[i].callback();
            return;
        }
    }
}