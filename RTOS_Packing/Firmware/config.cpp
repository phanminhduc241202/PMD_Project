#include "config.h"

volatile int emptySpotCounter = 0;
String currentUID = "";
Servo myservo1;
Servo myservo2;
LiquidCrystal_I2C lcd(0x27, 16, 2);
QueueHandle_t StatusQueue;
MFRC522 mfrc522(SS_PIN, RST_PIN);

struct User userList[NUMBER_OF_USER] = {
    {0, "43a2fefc", true},
    {1, "5345ef7", true},
    {2, "xxxxxxxx", false}
};

SystemState currentState = IDLE; // Da khoi tao trang thai ban dau cua he thong
