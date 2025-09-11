#pragma once

#include <Arduino.h>
#include "config.h"

unsigned char readStudentCount();
void writeStudentCount(unsigned char count);
void writeCardData(uint16_t startAddr, const unsigned char cardId[], size_t size);
void writeStudentId(uint16_t startAddr, const String& studentId);
bool findCardID(const String& rfidData, String& studentNo);
bool findStudentID(const String& studentID, String& studentNo);