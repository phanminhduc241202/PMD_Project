#include "LCDTask.h"

void DisplayLCD(void *pvParameters) {
  lcd.clear();
  bool sensorStatusRx[NUMBER_OF_SPOT];
  while (1) {
    emptySpotCounter = 0;
    if (xQueueReceive(StatusQueue, &sensorStatusRx, 0) == pdPASS) { // pdPASS la mot macro cua FreeRTOS, tra ve khi nhan data thanh cong
      lcd.setCursor(0, 0);
      lcd.print("Slots: ");
      for (int slot = 0; slot < NUMBER_OF_SPOT; slot++) {
        lcd.setCursor(slot, 1);
        if (!sensorStatusRx[slot]) {
          lcd.print('X');
        } else {
          emptySpotCounter++;
          lcd.print('O');
        }
      }
      lcd.setCursor(8, 0);
      lcd.print("Empty: ");
      lcd.print(emptySpotCounter);
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}