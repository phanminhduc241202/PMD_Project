# Electrical Equipment Control System Using Infrared Remote

A smart home automation system that enables control of electrical devices through infrared remote control and physical buttons, built with ESP32 microcontroller.

## Overview

This project implements an electrical equipment control system designed for home automation applications. The system allows users to control up to 4 electrical devices using either an infrared remote control or physical push buttons, with real-time status display and time tracking capabilities.

## Features

- **Dual Control Methods**: Control devices via infrared remote or physical buttons
- **Real-Time Display**: 16x2 LCD shows current time, date, and device status
- **4-Channel Control**: Simultaneous control of up to 4 electrical devices
- **Time Management**: Real-time clock with battery backup for accurate timekeeping
- **I2C Communication**: Efficient communication protocol minimizing pin usage
- **Low Power Design**: Optimized for energy efficiency

## Hardware Components

### Core Components
- **ESP32-WROM-32**: Main microcontroller (38-pin development board)
- **4-Channel Relay Module (5V)**: Device switching control
- **DS1307 RTC Module**: Real-time clock with battery backup
- **16x2 LCD with I2C Interface**: Status display
- **HX1838 Infrared Receiver**: Remote signal reception
- **Push Buttons (4x)**: Manual device control

### Power System
- **Input**: 12V DC adapter
- **Regulation**: LM7805 voltage regulator (12V to 5V)
- **Total Power Consumption**: ~256mA at 5V

## System Architecture

```
[Infrared Remote] ──┐
                    │
[Push Buttons] ──────┼──► [ESP32] ──────┼──► [4-Channel Relay] ──► [Electrical Devices]
                    │     (Central       │
[DS1307 RTC] ───────┘     Processor)    └──► [16x2 LCD Display]
```

## Pin Configuration

### ESP32 Connections
- **GPIO16-19**: Relay control (IN1-IN4)
- **GPIO21**: I2C SDA (LCD & RTC)
- **GPIO22**: I2C SCL (LCD & RTC)
- **GPIO25, 33, 32, 35**: Push button inputs
- **GPIO34**: Infrared receiver signal input

### Communication Protocols
- **I2C**: Real-time clock and LCD display
- **Digital I/O**: Relay control and button inputs
- **Infrared**: 38kHz carrier frequency for remote control

## Remote Control Functions

| Button | Function |
|--------|----------|
| 1-4 | Turn ON Relays 1-4 |
| 5-8 | Turn OFF Relays 1-4 |
| 9 | Display current time |
| * | Show date and time |
| # | Turn OFF all relays |

## Software Features

### Core Functionality
- Infrared signal decoding and processing
- Real-time clock management
- I2C communication handling
- Multi-device control logic
- LCD status display updates

### Libraries Used
- `IRremote`: Infrared signal processing
- `LiquidCrystal_I2C`: LCD display control
- `virtuabotixRTC`: Real-time clock management
- `Wire`: I2C communication

## Installation & Setup

### Hardware Assembly
1. Connect all components according to the pin configuration
2. Ensure proper power supply connections (12V input, 5V regulated)
3. Install backup battery in DS1307 RTC module

### Software Setup
1. Install Arduino IDE
2. Add ESP32 board support
3. Install required libraries:
   ```
   - IRremote
   - LiquidCrystal_I2C
   - virtuabotixRTC
   ```
4. Upload the provided code to ESP32

### Initial Configuration
- Set initial date/time in the code before first use
- Verify infrared remote button mappings
- Test all relay channels and buttons

## Usage

1. **Power On**: System initializes and displays startup message
2. **Manual Control**: Press physical buttons 1-4 to toggle corresponding relays
3. **Remote Control**: Use infrared remote for wireless device control
4. **Status Monitoring**: LCD shows real-time device status and time
5. **Time Display**: Press '*' on remote to view current date/time

## Technical Specifications

| Parameter | Value |
|-----------|--------|
| Operating Voltage | 5V DC |
| Power Consumption | 256mA |
| Communication | I2C, Digital I/O |
| Relay Switching | 5V, up to 10A per channel |
| Display | 16x2 Character LCD |
| IR Frequency | 38kHz |
| Operating Temperature | -25°C to 85°C |

## Future Enhancements

### Planned Features
- WiFi connectivity for remote monitoring
- Mobile app integration
- Advanced timer functionality
- Voice control integration
- Energy consumption monitoring

### Potential Improvements
- Enhanced IR signal processing
- Improved button debouncing
- PCB miniaturization
- Weather-resistant enclosure

## Known Limitations

- Susceptible to IR interference in bright environments
- Limited to 4-device control in current design
- Requires line-of-sight for infrared operation
- Manual time setting required after power loss

## Academic Context

This project was developed as part of the Computer Engineering Technology program at Ho Chi Minh City University of Technology and Education. It demonstrates practical application of:
- Embedded systems design
- Microcontroller programming
- Communication protocols
- Real-time systems
- Home automation concepts

