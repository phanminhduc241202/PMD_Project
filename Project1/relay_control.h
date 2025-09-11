
#ifndef RELAY_CONTROL_H
#define RELAY_CONTROL_H

#include "config.h"
#include <Arduino.h>

// Mảng trạng thái của các relay, được khai báo extern để các file khác có thể truy cập
extern int relay_states[NUM_DEVICES];

// Khai báo các hàm điều khiển relay
void initRelays();
void control_relay(int index);
void turn_off_all_relays();

#endif