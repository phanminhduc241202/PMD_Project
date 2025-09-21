#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino_FreeRTOS.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <queue.h>

#define RST_PIN 9
#define SS_PIN 10
#define BUZZER_PIN A0

#define NUMBER_OF_SPOT 6
#define NUMBER_OF_USER 3

// Khai báo cấu trúc
struct User {
  int id;
  String rfid;
  boolean status;
};

// Khai báo enum cho trạng thái hệ thống
enum SystemState {
  IDLE,
  CHECKIN,
  CHECKOUT
};

// Khai báo biến toàn cục
extern volatile int emptySpotCounter; // Biến đếm số chỗ trống, volatile để tránh tối ưu hóa bởi trình biên dịch
extern String currentUID; // Biến lưu trữ UID hiện tại
extern Servo myservo1;
extern Servo myservo2;
extern LiquidCrystal_I2C lcd;
extern QueueHandle_t StatusQueue; //QueueHandle_t là một biến con trỏ dùng để tham chiếu đến hàng đợi của thư viện FreeRTOS
extern MFRC522 mfrc522;
extern struct User userList[NUMBER_OF_USER];
extern SystemState currentState; // Biến lưu trữ trạng thái hiện tại của hệ thống

#endif
