#include "IO_Control.h"

void handleCheckin() {
  if (currentState == IDLE) {
    currentState = CHECKIN;
    digitalWrite(BUZZER_PIN, HIGH);
    vTaskDelay(100 / portTICK_PERIOD_MS);
    digitalWrite(BUZZER_PIN, LOW);
    for (int pos = 0; pos <= 90; pos += 1) {
      myservo1.write(pos);
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    currentState = IDLE;
  }
}

void handleCheckout() {
  if (currentState == IDLE) {
    currentState = CHECKOUT;
    digitalWrite(BUZZER_PIN, HIGH);
    vTaskDelay(100 / portTICK_PERIOD_MS);
    digitalWrite(BUZZER_PIN, LOW);
    for (int pos = 0; pos <= 90; pos += 1) {
      myservo2.write(pos);
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    currentState = IDLE;
  }
}

void handleInvalidCard() {
  if (currentState == IDLE) {
    currentState = IDLE;
    digitalWrite(BUZZER_PIN, HIGH);
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    digitalWrite(BUZZER_PIN, LOW);
  }
}