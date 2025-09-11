#include "RFIDTask.h"
#include "IO_Control.h"

void HandleRFID(void *pvParameters) {
  while (1) {
    if (mfrc522.PICC_IsNewCardPresent()) { // kiem tra xem co the moi gan dau doc the khong
      if (mfrc522.PICC_ReadCardSerial()) { // Doc serial UID cua the
        currentUID = "";
        for (byte i = 0; i < mfrc522.uid.size; i++) {
          currentUID += String(mfrc522.uid.uidByte[i], HEX); // chuyen UID tu byte sang hex
        }

        int userIndex = -1;
        for (int i = 0; i < NUMBER_OF_USER; i++) {
          if (userList[i].rfid.equals(currentUID)) { // so sanh UID vua doc voi user
            userIndex = i; // luu lai chi so nguoi dung neu khop
            break;
          }
        }

        if (userIndex != -1) { 
          if (userList[userIndex].status == false) { // kiem tra da vao trc do chua
            userList[userIndex].status = true; // cap nhat trang thai vao
            handleCheckin(); // goi servo cong vao
          } else {
            userList[userIndex].status = false; // cap nhat trang thai ra
            handleCheckout();
          }
        } else {
          handleInvalidCard(); // canh bao buzzer keo dai neu user khong ton tai
        }

        mfrc522.PICC_HaltA();  // dung qua trinh doc the
      }
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}