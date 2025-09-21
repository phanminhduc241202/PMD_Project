#include "StateManager.h"
#include "config.h"
#include "UI.h"
#include "RFIDManager.h"
#include "DataStorage.h"
#include "Alerts.h"

AppState currentState;
String idBuffer;
unsigned char cardIdArray[4];

void setAppState(AppState newState) {
  currentState = newState;
}

AppState getAppState() {
  return currentState;
}

void handleState() {
  char customKey = customKeypad.getKey();
  
  switch (currentState) {
    case AppState::MAIN_MENU:
      if (customKey == 'A') {
        setAppState(AppState::REG_MODE);
        showRegistrationScreen();
      } else if (customKey == 'B') {
        setAppState(AppState::SCAN_MODE);
        showScanScreen();
      } else if (customKey == 'C') {
        setAppState(AppState::ID_MODE);
        showIDScreen();
      }
      break;

    case AppState::REG_MODE:
      if (isNewCardPresent()) {
        getCardUID();
        setAppState(AppState::REG_WAIT_ID);
        lcd.clear();
        lcd.print("Enter student ID");
        lcd.setCursor(0, 1);
        idBuffer = "";
      }
      break;
    
    case AppState::REG_WAIT_ID:
      if (customKey) {
        if (customKey == 'D') {
          if (idBuffer.length() == 8) {
            // Write data to EEPROM
            unsigned char studentStored = readStudentCount();
            uint16_t studentIndexAddr = (studentStored == 0) ? 1 : studentStored * 13 + 1;
            
            // write student count
            writeStudentCount(studentStored + 1);

            // write student number
            mem.write(studentIndexAddr, studentStored + 1);
            
            // write card ID
            for(byte i = 0; i < mfrc522.uid.size; i++){
              mem.write(studentIndexAddr + 1 + i, mfrc522.uid.uidByte[i]);
            }
            
            // write student ID
            writeStudentId(studentIndexAddr + 5, idBuffer);

            showSuccessfulMessage();
            setAppState(AppState::MAIN_MENU);
            showMainMenu();

          } else {
            showInvalidIDMessage();
            delay(1000);
            setAppState(AppState::MAIN_MENU);
            showMainMenu();
          }
          idBuffer = "";
        } else {
          idBuffer += customKey;
          lcd.print(customKey);
        }
      }
      break;

    case AppState::SCAN_MODE:
      if (isNewCardPresent()) {
        String rfidData = getCardUID();
        String studentNo;
        if (findCardID(rfidData, studentNo)) {
          successAlert();
          lcd.clear();
          lcd.print("Student #" + studentNo);
          delay(2000);
        } else {
          failureAlert();
          showInvalidCardMessage();
          delay(1000);
        }
        setAppState(AppState::MAIN_MENU);
        showMainMenu();
      }
      break;

    case AppState::ID_MODE:
      if (customKey) {
        idBuffer += customKey;
        lcd.print(customKey);
      }
      if (customKey == 'D') {
        if (idBuffer.length() != 8) {
          showInvalidIDMessage();
          delay(1000);
          setAppState(AppState::MAIN_MENU);
          showMainMenu();
        } else {
          String studentNo;
          if (findStudentID(idBuffer, studentNo)) {
            successAlert();
            lcd.clear();
            lcd.print("Student #" + studentNo);
            delay(2000);
          } else {
            failureAlert();
            showInvalidIDMessage();
            delay(1000);
          }
          idBuffer = "";
          setAppState(AppState::MAIN_MENU);
          showMainMenu();
        }
      }
      break;
  }
}
