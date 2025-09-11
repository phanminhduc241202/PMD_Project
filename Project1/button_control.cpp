#include "button_control.h"
#include "relay_control.h"

void initButtons() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        pinMode(BUTTON_PINS[i], INPUT_PULLUP); // tin hieu dien tro keo de button o muc high
    }
}

void read_button() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        if (digitalRead(BUTTON_PINS[i]) == LOW) {
            delay(200);
            control_relay(i);
            break;
        }
    }
}