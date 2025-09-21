# Student Attendance Checking System

## Overview
This repository contains the technical design document and implementation details for an automated student attendance checking system. The system utilizes RFID technology for student identification, a keypad for input, and EEPROM for data storage to streamline the attendance process in educational settings. It was developed as a design project by students from the Computer Engineering Technology program at HCMUTE (Ho Chi Minh City University of Technology and Education) under the supervision of Assoc. Prof. Phan Van Ca.

The project addresses the inefficiencies of manual attendance checking, such as time consumption and inaccuracy, by providing an automated solution that saves time, enhances accuracy, and supports future scalability.

**Project Details:**
- **Project Name**: Student attendance checking system
- **Students**:
  - Trần Nam Phát (ID: 21119318)
  - Nguyễn Thành Giang (ID: 21119304)
  - Phan Minh Đức (ID: 21119303)
- **Major**: Computer Engineering Technology
- **Supervisor**: Assoc. Prof. Phan Van Ca
- **Copyright**: ©2021 CCE Department, HCMUTE
- **Confidentiality**: Confidential Property of CCE Dept.

## Problem Statement
Traditional attendance methods are time-consuming, inaccurate, and inefficient, especially in large classes. This system automates the process using RFID cards for registration and verification, a keypad for input, and EEPROM for storing data to prevent loss during power outages. It can be scaled to use a web server and database for centralized management.

## Objectives (Target)
- Design, develop, and implement an automated attendance system using RFID for identification and data storage.
- Streamline attendance checking, enhance efficiency, and ensure accurate records for lecturers and students.

## Content
The system provides automated student registration, attendance verification, and secure data storage. It addresses inefficiencies by leveraging RFID for quick identification, offering real-time data, improved accuracy, and support for data-driven decisions in education.

## Technical Objectives and Specifications
### Customers' Needs
- Automate attendance to save time, improve accuracy, provide real-time data, and enhance management.
- Support data-driven decision-making for better teaching and learning.

### Engineering Requirements
#### Functions
- Student Registration: Enroll new students using RFID cards, capturing and storing information securely.
- Attendance Verification: Identify students via RFID swipes automatically and reliably.
- Data Storage: Store student data, including attendance records, securely against unauthorized access and loss.

#### Non-Functions
- Usability: User-friendly interface requiring minimal technical knowledge.
- Reliability: Consistent and accurate operation under normal conditions.
- Availability: Minimal downtime during designated times.
- Scalability: Designed for future integration with web servers and databases as user base grows.

## Concept/Technology
### 3.1 I2C Protocol
Inter-Integrated Circuit protocol for communication between devices.

### 3.2 SPI Protocol
Serial Peripheral Interface for high-speed data transfer.

### 3.3 AT24C256 EEPROM Module
Non-volatile memory for storing student data.

### 3.4 16x2 LCD with Integrated I2C Module
Display for user interface and messages.

### 3.5 Keypad 4x4
Input device for entering student IDs and selecting modes.

### 3.6 RFID Reader Module
MFRC522 for reading RFID cards.

### 3.7 Main Processing Unit
ESP32 development kit as the central microcontroller.

## System Architecture
- **Block Diagram**: ESP32 controls keypad, LCD, EEPROM, LEDs, buzzer, and RFID reader.
- **Functionality**: Handles registration, scanning, and ID entry modes.

## Detailed Design
### System Operation
Modes: Registry (A), Card Scan (B), ID Entry (C), Enter (D).

### Hardware Design
- Keypad block
- LCD display block
- EEPROM storage block
- LED alert block (red for invalid, green for valid)
- Buzzer alert block
- RFID reader block
- Main processing block (ESP32)
- Power source block
- Full system schematic
- Flowchart of the whole system

**Code Breakdown**:
- Libraries: AT24Cxx, MFRC522, LiquidCrystal_I2C, Keypad.
- User-defined functions for reading/writing EEPROM, detecting RFID, initializing components.
- Main loop handles keypad inputs for modes, RFID scanning, ID entry, and alerts.

**Flowcharts**:
- Flowchart for the whole system.

## Final Product
(Refer to the document for images and details.)

## Appendix
### 7.1 Division of Labor
| ID | Activity | Description | Deliverables/Checkpoints | Duration (Days) | People Resources | Predecessors |
|----|----------|-------------|--------------------------|-----------------|------------------|--------------|
| 1 | Study on data transmission protocol | I2C How to transmit data | | 1 | Giang | |
| 2 | Take Data from RFID Reader | Gather user’s check-in information | Program to identify user data | 1 | Đức | Users information |
| 3 | EEPROM configuration | Configure addresses to save data to EEPROM | address configuration | | Phát | Users information 2 |
| 3.1 | Read EEPROM | Create a function to read data from EEPROM | EEPROM reader program | 1 | Phát | Users information 3 |
| 3.2 | Write EEPROM | Create a function to write data from EEPROM | EEPROM writing program | 1 | Phát | Users information 3, 3.1 |
| 4 | LCD Control | Display student information | Display LCD program | 1 | Giang | Process RFID reader’s data 1 |
| 5 | Buzzer and LEDs Control | Signals when there are valid and invalid signals | The program reports signals from the buzzer and leds | 1 | Đức | Process RFID reader’s data 2 |
| 6 | Constructing circuit | Component identification, component placement | Build and test circuit | 2 | Phát | Purchasing hardware component 1, 2, 3, 4, 5 |
| 7 | Constructing model | Design and assemble small scale models for the systems | Small system model | 2 | Đức, Giang, Phát | purchase materials 1, 2, 4, 4, 5, 6 |
| 8 | Write report | Write a complete report about the system | Report | 1 | Đức, Giang, Phát | summarize all the completed parts 1, 2, 3, 4, 5, 6, 7 |

(Note: Table may have inconsistencies in the extracted text; refer to document for accuracy.)

### 7.2 Bill of Materials
| ID | Parts/Components | Amount | Price per Unit (VND) | Total (VND) |
|----|------------------|--------|----------------------|------------|
| 1 | ESP32 | 1 | 140,000 | 140,000 |
| 2 | 4x4 Keypad | 1 | 48,000 | 48,000 |
| 3 | LCD I2C 16x2 | 1 | 56,000 | 56,000 |
| 4 | RFID Reader | 1 | 28,000 | 28,000 |
| 5 | RFID Card | 3 | 4,000 | 12,000 |
| 6 | EEPROM | 1 | 110,000 | 110,000 |
| 7 | Buzzer | 1 | 3,000 | 3,000 |
| 8 | Led | 1 | 3,000 | 3,000 |

**Total Cost**: 400,000 VND (calculated from listed items).

### 7.3 Gantt Chart
(Refer to the document for the visual Gantt chart outlining project timelines.)

### 7.4 User Manual
- When started, LCD displays options: A.REG, B.SCAN, C.ID, D.ENTER.
- REG MODE (A): Scan card and enter ID to register.
- SCAN MODE (B): Scan card to check information.
- ID MODE (C): Enter ID to check information.
- ENTER (D): Submit entered value.

### 7.5 Complete Code
(Full code provided in the document; includes userdef.h and main.c for ESP32 with libraries for EEPROM, RFID, LCD, Keypad.)

## List of Tables & Figures
- Fig1: AT24C256 EEPROM
- Fig2: AT24C256 module
- Fig3: LCD module
- Fig4: I2C Serial Interface Adapter Module
- Fig5: 4x4 Keypad
- Fig6: 4x4 Keypad schematic
- Fig7: RFID reader module
- Fig8: ESP32 development kit
- Fig9: ESP32 pin configuration
- Fig10: ESP32 schematic
- Fig11: System’s block diagram
- Fig12: 4x4 keypad block
- Fig13: LCD display block
- Fig14: EEPROM storage block
- Fig15: LED alert block
- Fig16: Buzzer alert block
- Fig17: RFID reader block
- Fig18: Main processing block
- Fig19: Power source block
- Fig20: Full system schematic
- Fig21: Flowchart for the whole system

## How to Use
1. Clone the repository.
2. Upload code to ESP32 using Arduino IDE.
3. Assemble hardware as per diagrams.
4. Register students via REG mode.
5. Check attendance via SCAN or ID modes.

## Contributors
- Trần Nam Phát
- Nguyễn Thành Giang
- Phan Minh Đức

## License
MIT License. See [LICENSE](LICENSE) for details.

For full details, refer to [final.pdf](final.pdf). Community contributions welcome!