#include "RFIDManager.h"
#include "config.h"

bool isNewCardPresent() {
  return mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial();
}

String getCardUID() {
  String readerData;
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    char temp[3];
    sprintf(temp, "%02x", mfrc522.uid.uidByte[i]);
    readerData += String(temp);
  }
  return readerData;
}
