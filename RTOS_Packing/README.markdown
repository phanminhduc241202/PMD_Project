# Apartment Car Parking Management System for Pre-Registered Users

## Overview
This repository hosts the complete technical design document (TDD), source code, and supporting materials for a smart parking management system tailored for apartment buildings. The system is designed to handle vehicle access for pre-registered residents using RFID technology, infrared sensors for vacancy detection, and real-time operating system (RTOS) for efficient multitasking. Developed as a technical design project by students in the Computer Engineering Technology program at Ho Chi Minh City University of Technology and Education (HCMUTE), under the supervision of Assoc. Prof. Phan Van Ca.

The project tackles key challenges in apartment parking lots, including improper parking space allocation, security vulnerabilities like theft and sabotage, inefficient traffic flow, and inadequate surveillance of incoming/outgoing vehicles. By automating access control, vacancy monitoring, and user notifications, the system enhances security, optimizes space usage, and improves overall user convenience.

**Project Details:**
- **Project Name**: Management system in apartment car parking lot for pre-registered users
- **Students**:
  - [1] Trần Nam Phát (ID: 21119318)
  - [2] Phan Minh Đức (ID: 21119303)
  - [3] Nguyễn Thành Giang (ID: 21119304)
- **Major**: Computer Engineering Technology
- **Supervisor**: Assoc. Prof. Phan Van Ca
- **Copyright**: ©2021 CCE Department, HCMUTE
- **Confidentiality**: Confidential Property of CCE Dept.

## Terms and Abbreviations
- **[IOT]**: Internet of Things
- **[RFID]**: Radio Frequency Identification
- **[IDE]**: Integrated Development Environment
- **[RTOS]**: Real Time Operating System
- **[IR]**: Infrared

## Problem Statement
Urban apartment living, particularly among young people, has led to a surge in resident numbers, complicating vehicle management—especially cars. Issues include:
- Inappropriate parking space allocation causing confusion and improper parking.
- Blocked traffic flow within the parking lot.
- Poor labor division leading to theft and asset sabotage.
- Inadequate procedures for monitoring incoming/outgoing vehicles, risking user property.

The proposed system designs a car management solution to secure the parking lot, protect user assets, allocate spaces efficiently, control traffic flow, and manage user information for security.

## Objectives (Target)
- Ensure safety and protection of users' assets in the parking area.
- Provide real-time management and monitoring of parking space status and information.
- Optimize management capabilities and enhance user experience.
- Create a safe, efficient, and convenient environment for users.

## Content
The system manages vehicles in real-time, providing:
- Parking lot status information.
- Asset protection via pre-registered user data management.
- Allocation of incoming cars to vacant spots to reduce congestion and save time.
- Display of vacant spot positions to users.
- Detailed reports on parking status, historical logs, and related data.

## Technical Objectives and Specifications
### Customers' Needs
- Manage cars inside the parking lot.
- Support distributing vacant parking spaces for incoming cars.

### Engineering Requirements
#### Functions
- Use RFID reader to compare and verify user's registered information.
- Activate buzzer on detection of false check-out information.
- Extract data on vacant parking spaces from sensors and display to entering users.

#### Non-Functions
- Barrier for blocking vehicles during identification.
- Step motor (servo) for controlling barrier lifting/lowering.
- Bumps to hold vehicles in place for scanning.
- Power supply (anticipated 4 AA batteries, 6-12V for small model).

### Specifications
#### Data
- RFID card data from user.
- Data from sensors.

#### Range of Operation
- Infrared sensors in parking spaces to detect vacancies and inform users of vehicle locations.

#### Real-Time Requirements
- Delay for displaying vacant parking spots: 0.5s.
- Delay for opening/closing barrier: 1.5s.

#### Hardware
- Operation voltage:
  - Input for main board: 7-12V.
  - I/O voltage: 5V.

#### Software
- Arduino IDE, Visual Studio Code, Platform IO.

#### Communication and Connectivity
- I2C protocol.

## Concept/Technology
### Free RTOS
An open-source RTOS kernel for microcontrollers, providing real-time scheduling, inter-task communication, timing, and synchronization.

### Arduino Uno R3
Microcontroller board based on ATmega328P with 14 digital I/O pins (6 PWM), 6 analog inputs, 16 MHz resonator, USB connection, power jack, ICSP header, and reset button. User-friendly and low-cost for replacement.

### Infrared Sensor
Used for object detection in daily life and industry (e.g., TV remotes). Includes IR LED and photodiode forming a photo-coupler. Based on Planck's radiation, Stefan-Boltzmann, and Wien's displacement laws.

**Technical Specifications** (Using LM393 IC):
- Opening Angle: 35°
- Operating Voltage: 3.0V – 6.0V
- Detection Range: 2cm – 30cm (adjustable)
- Dimensions: 4.5cm (L) x 1.4cm (W) x 0.7cm (H)
- Output Logic: Low (obstacle present), High (no obstacle)
- Current Consumption: ~23 mA (3.3V), ~43 mA (5V)

### LCD Screen
Liquid crystal display module for various devices. Preferred for low cost, programmability, and custom character/animation support.

**Features**:
- Operating Voltage: 4.7V-5.3V
- 2 rows x 16 characters
- Current: 1mA (no backlight)
- 5x8 pixel characters
- Alphanumeric display
- Modes: 4-bit & 8-bit
- Backlights: Blue & Green

### Servo Motor
Actuator for discrete drive systems. Operates via electronic switches sending signals to stator. Rotor rotation corresponds to switching times, direction, and speed.

### RFID RC522
System with tag and reader. Reader generates high-frequency field; tag (passive) processes and transmits data via antenna.

### Buzzer
5 VDC buzzer for compact alarm circuits. Long life, stable performance.

**Specifications**:
- Power: 3.5V - 5.5V
- Current: <25mA
- Resonance Frequency: 2300Hz ± 500Hz
- Sound Amplitude: >80 dB
- Operating Temperature: -20°C to +70°C
- Dimensions: Diameter 12mm, height 9.7mm

## System Architecture
- **Block Diagram**: Arduino Uno controls RFID module, servos (entry/exit), infrared sensors, buzzer, and LCD (I2C).
- **3D Block Diagram**: Visual representation of components.
- **Circuit Diagrams**:
  - Connecting sensors, buzzer, and LCD.
  - Pinout for RC522 to Arduino.
- Description: RFID reads tags for verification; servos control barriers; sensors detect vacancies; buzzer alerts invalid actions; LCD displays info.

## Detailed Design
**Components**:
- MFRC522 RFID Reader
- Infrared Sensors
- Servo Motors
- Buzzer
- LCD Display (I2C)

**System Overview**:
- Manages parking spots, vehicle presence, and RFID-based check-in/out.

**Code Breakdown**:
- Libraries: Arduino_FreeRTOS, SPI, MFRC522, Servo, Wire, LiquidCrystal_I2C, queue.
- Definitions: Pins, spots (6), users (3), RFID codes.
- Setup: Initializes serial, RFID, servos, LCD, FreeRTOS tasks.
- Tasks: HandleRFID (card detection/operations), ReadSensor (sensor status queue), DisplayLCD (vacancy display).
- Functions: handleCheckin, handleCheckout, handleInvalidCard.

**Flowcharts**:
- Top-level Flowchart
- Task readSensor Flowchart
- Task Display LCD Flowchart
- Task handle RFID Flowchart

**Code**: Full C++ code provided in the document, using FreeRTOS for concurrency.

## Final Product
- LCD displaying empty spaces (e.g., 'O' for open, 'X' for occupied).
- Whole block model prototype.

**Instructions for Using the Car Parking System**:
1. **Register RFID**: Register card at management counter.
2. **Open Barrier**: Swipe RFID at entrance; barrier opens if valid, LCD shows "Barrier Opened."
3. **Invalid Card**: Alarm sounds, LCD shows "Invalid RFID Tag. Contact management."
4. **Check Vacancy**: View LCD for spots (e.g., "Vacancy: C3, C5, D7").
5. **Time Limit**: Barrier closes after a configured time.
6. **Contact Support**: For issues, contact management.
7. **Safety Notes**: Follow rules, secure RFID card.

## References
1. "RFID-based smart parking management system" – Eirini Eleni Tsiropoulou et al. (2017), Cyber-Physical Systems, DOI: 10.1080/23335777.2017.1358765.
2. "Car Park System: A Review of Smart Parking System and its Technology" – M.Y. Idris et al. (2009), Information Technology Journal, ISSN 1812-5638.
3. "Intelligent Parking Management System Based on Image Processing" – Hilal Al-Kharusi & Ibrahim Al-Bahadly (2014), World Journal of Engineering and Technology, DOI:10.4236/wjet.2014.22006.
4. "IOT based Smart Parking Management System" – J. Cynthia et al. (2018), International Journal of Recent Technology and Engineering, ISSN: 2277-3878.

