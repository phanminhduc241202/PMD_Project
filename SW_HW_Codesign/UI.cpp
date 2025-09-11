#include "UI.h"
#include "config.h"

void initLCD() {
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  showMainMenu();
}

void showMainMenu() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("A.REG   B.SCAN");
  lcd.setCursor(0, 1);
  lcd.print("C.ID    D.ENTER");
}

void showRegistrationScreen() {
  lcd.clear();
  lcd.print("REGISTRY MODE");
  delay(1000);
  lcd.clear();
  lcd.print("Scan card");
  lcd.setCursor(0, 1);
}

void showScanScreen() {
  lcd.clear();
  lcd.print("CARD MODE");
  delay(1000);
  lcd.clear();
  lcd.print("Scan card");
  lcd.setCursor(0, 1);
}

void showIDScreen() {
  lcd.clear();
  lcd.print("ID MODE");
  delay(1000);
  lcd.clear();
  lcd.print("Enter student ID");
  lcd.setCursor(0, 1);
}

void showTimeoutMessage() {
  lcd.clear();
  lcd.print("Time out");
  lcd.setCursor(0, 1);
  lcd.print("Retry");
  delay(1000);
  showMainMenu();
}

void showInvalidIDMessage() {
  lcd.clear();
  lcd.print("Invalid ID");
  lcd.setCursor(0, 1);
  lcd.print("Retry");
}

void showInvalidCardMessage() {
  lcd.clear();
  lcd.print("Invalid card ID");
  lcd.setCursor(0, 1);
  lcd.print("Retry");
}

void showSuccessfulMessage() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Successful");
  delay(1000);
}

void printLCD(const String& message) {
  lcd.print(message);
}