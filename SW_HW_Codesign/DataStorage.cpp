#include "DataStorage.h"

unsigned char readStudentCount() { // Doc tong so luong sinh vien
  return mem.read(0);
}

void writeStudentCount(unsigned char count) { // Ghi tong so luong sinh vien
  mem.write(0, count);
}

void writeCardData(uint16_t startAddr, const unsigned char cardId[], size_t size) {
  for (size_t i = 0; i < size; i++) {
    mem.write(startAddr + i, cardId[i]);
  }
} // ham ghi du lieu the RFID

void writeStudentId(uint16_t startAddr, const String& studentId) {
  for (int i = 0; i < studentId.length(); i++) {
    mem.write(startAddr + i, (uint8_t)studentId[i]);
  }
} // Ham ghi du lieu ID sinh vien

bool findCardID(const String& rfidData, String& studentNo) {
  for (uint16_t i = 1; i <= 1000; i += 13) {
    String eepromData = "";
    for (int j = 1; j <= 4; j++) {
      char data[3];
      sprintf(data, "%x", mem.read(i + j));
      eepromData += String(data);
    }
    if (rfidData == eepromData) {
      studentNo = String(mem.read(i));
      return true;
    }
  }
  return false;
} // Ham tim kiem ID the RFID

bool findStudentID(const String& studentID, String& studentNo) {
  for (uint16_t i = 1; i <= 1000; i += 13) {
    String eepromData = "";
    for (int j = 5; j <= 12; j++) {
      char data[2];
      sprintf(data, "%c", mem.read(i + j));
      eepromData += String(data);
    }
    if (studentID == eepromData) {
      studentNo = String(mem.read(i));
      return true;
    }
  }
  return false;
} // Ham tim kiem ID sinh vien