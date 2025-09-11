#include "relay_control.h"
#include "config.h"

int relay_states[NUM_DEVICES] = {HIGH, HIGH, HIGH, HIGH}; // HIGH = OFF // Relay hoat dong muc thap

void initRelays() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        pinMode(RELAY_PINS[i], OUTPUT);
        digitalWrite(RELAY_PINS[i], HIGH);
    }
}

void control_relay(int index) {
    if (index >= 0 && index < NUM_DEVICES) {
        relay_states[index] = !relay_states[index]; // dao trang thai relay
        digitalWrite(RELAY_PINS[index], relay_states[index]);
        delay(50);
    }
}

void turn_off_all_relays() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        digitalWrite(RELAY_PINS[i], HIGH);
        relay_states[i] = HIGH;
    }
}

