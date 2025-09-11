#include "sensorTask.h"

void ReadSensor(void *pvParameters) {
  bool sensorStatusTx[NUMBER_OF_SPOT + 2];
  while (1) {
    for (int i = 2; i < NUMBER_OF_SPOT + 2; i++) {
      int val = digitalRead(i);
      sensorStatusTx[i - 2] = val;
    }
    xQueueSend(StatusQueue, &sensorStatusTx, 0);
    // gui du lieu trang thai cam bien vao hang doi
    // 0 neu hang doi day se ko doi nua ma skip qua lenh
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}