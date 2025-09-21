#include <Arduino_FreeRTOS.h>
#include "config.h"
#include "RFIDTask.h"
#include "sensorTask.h"
#include "LCDTask.h"
#include "IO_Control.h"

// Bien toan cuc
volatile int emptySpotCounter;
String currentUID = "";
Servo myservo1;
Servo myservo2;
LiquidCrystal_I2C lcd(0x27, 16, 2);
QueueHandle_t StatusQueue;

void setup() {
  Serial.begin(9600);
  SPI.begin(); // Khoi tao giao tiep SPI cho RFID
  mfrc522.PCD_Init(); // Khoi tao RFID
  myservo1.attach(A1);
  myservo2.attach(A2);
  Serial.println("System Initialized");
  pinMode(BUZZER_PIN, OUTPUT);
  lcd.init();
  lcd.backlight();

  // Khoi tao hang doi (queue)
  StatusQueue = xQueueCreate(NUMBER_OF_SPOT + 2, sizeof(bool) * (NUMBER_OF_SPOT + 2));
  if (StatusQueue == NULL) {
    // Xu ly loi neu khong tao duoc hang doi
    Serial.println("Error: Could not create queue");
  }

  // Tao cac tac vu
  xTaskCreate(ReadSensor, "ReadSensor", 128, NULL, 1, NULL);
  xTaskCreate(HandleRFID, "HandleRFID", 128, NULL, 2, NULL);
  xTaskCreate(DisplayLCD, "DisplayLCD", 128, NULL, 1, NULL);

  vTaskStartScheduler(); // Bat dau lap lich he thong FreeRTOS
}

void loop() {
  // FreeRTOS se quan ly cac tac vu, nen loop() se trong
}
