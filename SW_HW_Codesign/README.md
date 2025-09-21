# Student Attendance Checking System

## Overview
This repository contains the technical design document and implementation details for an automated student attendance checking system. The system utilizes RFID technology for student identification, a keypad for input, and EEPROM for data storage to streamline the attendance process in educational settings. It was developed as a design project by students from the Computer Engineering Technology program at HCMUTE (Ho Chi Minh City University of Technology and Education) under the supervision of Assoc. Prof. Phan Van Ca.

The project addresses the inefficiencies of manual attendance checking, such as time consumption and inaccuracy, by providing an automated solution that saves time, enhances accuracy, and supports future scalability.

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
### I2C Protocol
Inter-Integrated Circuit protocol for communication between devices.

### SPI Protocol
Serial Peripheral Interface for high-speed data transfer.

### AT24C256 EEPROM Module
Non-volatile memory for storing student data.

### 16x2 LCD with Integrated I2C Module
Display for user interface and messages.

### Keypad 4x4
Input device for entering student IDs and selecting modes.

### RFID Reader Module
MFRC522 for reading RFID cards.

### Main Processing Unit
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

### User Manual
- When started, LCD displays options: A.REG, B.SCAN, C.ID, D.ENTER.
- REG MODE (A): Scan card and enter ID to register.
- SCAN MODE (B): Scan card to check information.
- ID MODE (C): Enter ID to check information.
- ENTER (D): Submit entered value.

